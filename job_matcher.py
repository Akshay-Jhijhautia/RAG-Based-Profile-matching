"""Match a job description against the resumes indexed by ``resume_rag``.

The job description is parsed for required skills and must-have requirements
("5+ years of Python", "Master's degree"), embedded with the same embedder
that built the index, and compared against every resume chunk. Chunk hits are
grouped per candidate to give the top-K most similar resumes.
"""

from __future__ import annotations

import re

import resume_rag
import skills_vocab

TOP_K = 10
# Chunks retrieved per query before grouping by candidate (about 4-5 per resume).
CHUNK_POOL = 50
# Candidate semantic score = blend of the best chunk and the mean of the top 3,
# so one strong section counts most but broad relevance still helps.
BEST_CHUNK_WEIGHT = 0.6
TOP_CHUNKS_FOR_MEAN = 3

_BULLET_RE = re.compile(r"^\s*[-•*▪◦·]\s*")
_PREFERRED_RE = re.compile(
    r"\b(?:nice[- ]to[- ]haves?|preferred|bonus|a plus|desirable|good[- ]to[- ]haves?|optional)\b",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(
    r"\b(?:requirements?|required|qualifications|responsibilities|about|what you'?ll|"
    r"who you are|what we'?re looking for|must[- ]haves?|the role|benefits|perks|we offer)\b",
    re.IGNORECASE,
)
# "5+ years", "5 + yrs", "3-5 years", "at least 4 years of hands-on experience with"
_YEARS_RE = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\s*(?:\+|(?:-|–|to)\s*\d+(?:\.\d+)?)?\s*\+?\s*(?:years?|yrs?)\b"
    r"(?:\s+of)?(?:\s+(?:professional|hands-on|industry|commercial|relevant|production))?"
    r"(?:\s+experience)?(?:\s+(?:in|with|using|building|developing|of))?\s*(?P<rest>[^.;\n]{0,80})",
    re.IGNORECASE,
)
# A skill must follow a years phrase this closely to be the subject of it.
_YEARS_SKILL_WINDOW = 25


def _is_jd_header(line: str) -> bool:
    """A heading such as "Requirements:" or "Nice to have" (not a short requirement line)."""
    stripped = line.strip()
    if not stripped or _BULLET_RE.match(line) or len(stripped) > 60:
        return False
    if stripped.endswith(":"):
        return True
    return len(stripped.split()) <= 5 and bool(
        _SECTION_HEADING_RE.search(stripped) or _PREFERRED_RE.search(stripped)
    )


def _years_requirements(line: str) -> list[dict]:
    requirements = []
    for match in _YEARS_RE.finditer(line):
        years = float(match["years"])
        named = skills_vocab.find_skills(match["rest"][:_YEARS_SKILL_WINDOW])
        source = match.group(0).strip()
        if named:
            requirements.append(
                {"type": "skill_years", "skill": named[0], "min_years": years, "source": source}
            )
        elif "experience" in line.lower():
            requirements.append({"type": "total_years", "min_years": years, "source": source})
    return requirements


def parse_job_description(job_description: str) -> dict:
    """Pull skills and must-have requirements out of a job description.

    Lines under a "Nice to have" / "Preferred" heading, and clauses (split on
    ";") that say "preferred" / "a plus" themselves, are optional; everything
    else is core.
    Must-haves come from core lines only:

    - ``{"type": "skill_years", "skill", "min_years"}`` from "5+ years of Python"
    - ``{"type": "total_years", "min_years"}`` from "3+ years of experience"
    - ``{"type": "degree", "degree"}`` from "Master's degree in CS" (the lowest
      degree named on the line, so "Bachelor's or Master's" means Bachelor's)

    Each must-have also carries the ``source`` text it was parsed from.
    Returns ``{"text", "required_skills", "preferred_skills", "must_haves"}``.
    """
    core_lines, optional_lines = [], []
    in_optional_section = False
    for line in job_description.splitlines():
        if _is_jd_header(line):
            in_optional_section = bool(_PREFERRED_RE.search(line))
        if in_optional_section:
            optional_lines.append(line)
            continue
        for clause in line.split(";"):
            (optional_lines if _PREFERRED_RE.search(clause) else core_lines).append(clause)

    required = skills_vocab.find_skills("\n".join(core_lines))
    preferred = [s for s in skills_vocab.find_skills("\n".join(optional_lines)) if s not in required]

    must_haves = []
    for line in core_lines:
        must_haves.extend(_years_requirements(line))
        degrees = resume_rag.degrees_mentioned(line)
        if degrees:
            must_haves.append({"type": "degree", "degree": degrees[-1], "source": line.strip()})

    return {
        "text": job_description.strip(),
        "required_skills": required,
        "preferred_skills": preferred,
        "must_haves": _dedupe_must_haves(must_haves),
    }


def _dedupe_must_haves(must_haves: list[dict]) -> list[dict]:
    """Keep the strictest requirement of each kind (per skill for skill years)."""
    strictest: dict[tuple, dict] = {}
    for req in must_haves:
        key = (req["type"], req.get("skill"))
        current = strictest.get(key)
        if current is None:
            strictest[key] = req
        elif req["type"] == "degree":
            if resume_rag.DEGREE_LEVELS[req["degree"]] > resume_rag.DEGREE_LEVELS[current["degree"]]:
                strictest[key] = req
        elif req["min_years"] > current["min_years"]:
            strictest[key] = req
    return list(strictest.values())


def _excerpt(document: str) -> str:
    """Chunk text without the ``Candidate: ... | Section: ...`` embedding prefix."""
    return document.split("\n", 1)[1] if "\n" in document else document


class JobMatcher:
    """Finds the resumes in a ``ResumeRAG`` index that best match a job description."""

    def __init__(self, rag: resume_rag.ResumeRAG):
        self.rag = rag

    def semantic_search(self, job_description: str, k: int = TOP_K) -> list[dict]:
        """Top-``k`` candidates by embedding similarity to the job description.

        The JD is embedded and the ``CHUNK_POOL`` nearest chunks are fetched
        from Chroma; chunk similarity is ``1 - cosine distance``. Chunks are
        grouped per resume and scored as
        ``BEST_CHUNK_WEIGHT * best + (1 - BEST_CHUNK_WEIGHT) * mean(top 3)``.

        Returns ``[{"resume_path", "candidate_name", "metadata",
        "semantic_score", "chunks"}]`` best first, where ``chunks`` lists the
        candidate's matched chunks (``id``, ``section``, ``text``, ``similarity``)
        best first.
        """
        if not job_description or not job_description.strip():
            raise ValueError("Job description must be a non-empty string")
        total_chunks = self.rag.collection.count()
        if total_chunks == 0:
            return []

        (query,) = self.rag.embedder.embed([job_description])
        result = self.rag.collection.query(
            query_embeddings=[query],
            n_results=min(CHUNK_POOL, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        candidates: dict[str, dict] = {}
        for chunk_id, document, record, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            path = record["resume_path"]
            if path not in candidates:
                candidates[path] = {
                    "resume_path": path,
                    "candidate_name": record["candidate_name"],
                    "metadata": resume_rag.parse_record(record),
                    "chunks": [],
                }
            candidates[path]["chunks"].append(
                {
                    "id": chunk_id,
                    "section": record["section"],
                    "text": _excerpt(document),
                    "similarity": round(1.0 - distance, 4),
                }
            )

        for candidate in candidates.values():
            sims = sorted((c["similarity"] for c in candidate["chunks"]), reverse=True)
            top = sims[:TOP_CHUNKS_FOR_MEAN]
            score = BEST_CHUNK_WEIGHT * sims[0] + (1 - BEST_CHUNK_WEIGHT) * sum(top) / len(top)
            candidate["semantic_score"] = round(score, 4)
            candidate["chunks"].sort(key=lambda c: c["similarity"], reverse=True)

        ranked = sorted(candidates.values(), key=lambda c: c["semantic_score"], reverse=True)
        return ranked[:k]
