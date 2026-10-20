"""Match a job description against the resumes indexed by ``resume_rag``.

The job description is parsed for required skills and must-have requirements
("5+ years of Python", "Master's degree"). Candidates are then found with a
hybrid search:

- semantic: the JD is embedded with the same embedder that built the index
  and compared against every resume chunk; hits are grouped per candidate;
- keyword: BM25 over each resume's text, with canonical skill tokens added so
  "ML" in a JD matches "machine learning" in a resume;
- critical skills: the JD's required skills checked against each candidate's
  extracted skills.

The candidate pool is the union of the semantic and keyword hits, so a resume
with an exact critical skill is not missed when its embedding ranks low.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

import resume_rag
import skills_vocab

TOP_K = 10
# Resumes taken from the keyword ranking into the hybrid pool.
KEYWORD_POOL = 20
# Hybrid score weights (sum to 1). Without required skills in the JD, the skill
# weight is spread over the other two in proportion.
SEMANTIC_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.2
SKILL_WEIGHT = 0.3
# Query repeats per skill token: required skills count double in BM25.
REQUIRED_SKILL_BOOST = 2
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


# --- keyword search ------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")


def _skill_token(skill: str) -> str:
    return "skill:" + skill.lower()


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stopwords, keeping "c++", "c#" and "node.js" whole."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in resume_rag.STOPWORDS]


def _resume_tokens(texts: list[str], skills: list[str]) -> list[str]:
    """BM25 document for one resume: its words plus one token per extracted skill."""
    tokens = [t for text in texts for t in tokenize(text)]
    return tokens + [_skill_token(s) for s in skills]


def _query_tokens(parsed: dict) -> list[str]:
    tokens = tokenize(parsed["text"])
    tokens += [_skill_token(s) for s in parsed["required_skills"]] * REQUIRED_SKILL_BOOST
    tokens += [_skill_token(s) for s in parsed["preferred_skills"]]
    return tokens


def _skill_match(required: list[str], preferred: list[str], candidate_skills: list[str]) -> dict:
    have = {s.lower() for s in candidate_skills}
    matched = [s for s in required if s.lower() in have]
    return {
        "matched_skills": matched,
        "missing_skills": [s for s in required if s.lower() not in have],
        "matched_preferred_skills": [s for s in preferred if s.lower() in have],
        "skill_coverage": round(len(matched) / len(required), 4) if required else None,
    }


def hybrid_score(semantic: float, keyword: float, skill_coverage: float | None) -> float:
    """Weighted blend of the three signals, each in 0..1."""
    if skill_coverage is None:
        total = SEMANTIC_WEIGHT + KEYWORD_WEIGHT
        return (SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword) / total
    return SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword + SKILL_WEIGHT * skill_coverage


class JobMatcher:
    """Finds the resumes in a ``ResumeRAG`` index that best match a job description."""

    def __init__(self, rag: resume_rag.ResumeRAG):
        self.rag = rag

    # --- semantic ---

    def _embed_query(self, job_description: str) -> list[float]:
        if not job_description or not job_description.strip():
            raise ValueError("Job description must be a non-empty string")
        (query,) = self.rag.embedder.embed([job_description])
        return query

    def _semantic_candidates(
        self, query: list[float], resume_paths: list[str] | None = None
    ) -> dict[str, dict]:
        """Nearest chunks grouped by resume, optionally restricted to ``resume_paths``.

        Chunk similarity is ``1 - cosine distance``; a candidate's
        ``semantic_score`` is ``BEST_CHUNK_WEIGHT * best + (1 -
        BEST_CHUNK_WEIGHT) * mean(top 3)``.
        """
        total_chunks = self.rag.collection.count()
        if total_chunks == 0 or resume_paths == []:
            return {}
        where = {"resume_path": {"$in": resume_paths}} if resume_paths else None
        result = self.rag.collection.query(
            query_embeddings=[query],
            n_results=min(CHUNK_POOL, total_chunks),
            where=where,
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
            candidate["chunks"].sort(key=lambda c: c["similarity"], reverse=True)
            sims = [c["similarity"] for c in candidate["chunks"]]
            top = sims[:TOP_CHUNKS_FOR_MEAN]
            score = BEST_CHUNK_WEIGHT * sims[0] + (1 - BEST_CHUNK_WEIGHT) * sum(top) / len(top)
            candidate["semantic_score"] = round(score, 4)
        return candidates

    def semantic_search(self, job_description: str, k: int = TOP_K) -> list[dict]:
        """Top-``k`` candidates by embedding similarity alone.

        Returns ``[{"resume_path", "candidate_name", "metadata",
        "semantic_score", "chunks"}]`` best first, where ``chunks`` lists the
        candidate's matched chunks (``id``, ``section``, ``text``, ``similarity``)
        best first.
        """
        query = self._embed_query(job_description)
        candidates = self._semantic_candidates(query)
        ranked = sorted(candidates.values(), key=lambda c: c["semantic_score"], reverse=True)
        return ranked[:k]

    # --- keyword ---

    def keyword_scores(self, parsed: dict) -> dict[str, float]:
        """BM25 score per resume path for a parsed JD, scaled so the best resume is 1.0."""
        texts: dict[str, list[str]] = {}
        skills: dict[str, list[str]] = {}
        for chunk in self.rag.all_chunks():
            path = chunk["metadata"]["resume_path"]
            texts.setdefault(path, []).append(_excerpt(chunk["document"]))
            skills.setdefault(path, list(chunk["metadata"].get("skills", [])))
        if not texts:
            return {}

        paths = list(texts)
        bm25 = BM25Okapi([_resume_tokens(texts[p], skills[p]) for p in paths])
        raw = bm25.get_scores(_query_tokens(parsed))
        best = max(raw)
        if best <= 0:
            return {p: 0.0 for p in paths}
        return {p: round(max(0.0, float(score)) / best, 4) for p, score in zip(paths, raw)}

    # --- hybrid ---

    def hybrid_search(
        self, job_description: str, k: int = TOP_K, parsed: dict | None = None
    ) -> list[dict]:
        """Top-``k`` candidates by the hybrid of semantic, keyword and skill signals.

        The pool is every candidate in the semantic chunk pool plus the
        ``KEYWORD_POOL`` best keyword matches; keyword-only candidates get
        their semantic score from a query restricted to their own chunks.

        Each result extends the ``semantic_search`` fields with
        ``keyword_score``, ``matched_skills``, ``missing_skills``,
        ``matched_preferred_skills``, ``skill_coverage`` (``None`` when the JD
        names no skills), ``hybrid_score`` (0..1) and, per chunk,
        ``skills`` (the JD skills that chunk mentions).
        """
        parsed = parsed or parse_job_description(job_description)
        query = self._embed_query(job_description)
        candidates = self._semantic_candidates(query)

        keyword = self.keyword_scores(parsed)
        by_keyword = sorted((p for p in keyword if keyword[p] > 0), key=keyword.get, reverse=True)
        keyword_only = [p for p in by_keyword[:KEYWORD_POOL] if p not in candidates]
        candidates.update(self._semantic_candidates(query, keyword_only))

        jd_skills = parsed["required_skills"] + parsed["preferred_skills"]
        for candidate in candidates.values():
            candidate["keyword_score"] = keyword.get(candidate["resume_path"], 0.0)
            candidate.update(
                _skill_match(
                    parsed["required_skills"],
                    parsed["preferred_skills"],
                    candidate["metadata"]["skills"],
                )
            )
            for chunk in candidate["chunks"]:
                mentioned = set(skills_vocab.with_implied(skills_vocab.find_skills(chunk["text"])))
                chunk["skills"] = [s for s in jd_skills if s in mentioned]
            candidate["hybrid_score"] = round(
                hybrid_score(
                    candidate["semantic_score"],
                    candidate["keyword_score"],
                    candidate["skill_coverage"],
                ),
                4,
            )

        ranked = sorted(
            candidates.values(),
            key=lambda c: (c["hybrid_score"], c["semantic_score"]),
            reverse=True,
        )
        return ranked[:k]
