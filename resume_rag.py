"""RAG pipeline for resumes: load, chunk by section, extract metadata, embed, store.

Resumes are read with the Milestone 1 file system tools (``fs_tools``), split
into their natural sections (Summary, Skills, Experience, Education, ...) and
chunked so that a chunk never mixes two sections and, where possible, never
splits a single job entry. Each chunk is embedded and stored in a local
ChromaDB collection together with the resume's metadata (name, skills,
experience years, education), so the job matcher can filter and rank on it.

CLI:
    python resume_rag.py index [resumes/] [--embedder openai|hashing] [--force] [--reset]
    python resume_rag.py stats
    python resume_rag.py list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Protocol

import chromadb
import openai
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from dotenv import load_dotenv

import fs_tools
import skills_vocab

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "gpt-4o-mini"
LLM_MAX_INPUT_CHARS = 12000
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
DEFAULT_DB_DIR = "chroma_db"
COLLECTION_NAME = "resumes"
# Part of every resume's content hash: bump it when chunking or metadata
# extraction changes so the next `index` run refreshes every resume.
PIPELINE_VERSION = "2"

MAX_CHUNK_CHARS = 1500
HEADER_SECTION = "Header"
GENERAL_SECTION = "General"
# A preamble longer than name + contact lines carries real content worth embedding.
MAX_HEADER_LINES_TO_SKIP = 3

# Canonical section name -> header spellings (compared after normalization).
SECTION_ALIASES = {
    "Summary": [
        "summary", "professional summary", "career summary", "profile",
        "professional profile", "objective", "career objective", "about", "about me",
    ],
    "Skills": [
        "skills", "technical skills", "core skills", "key skills", "core competencies",
        "competencies", "technologies", "tech stack", "skills and technologies",
    ],
    "Experience": [
        "experience", "work experience", "professional experience", "employment",
        "employment history", "work history", "career history", "relevant experience",
    ],
    "Education": ["education", "academic background", "education and training", "qualifications"],
    "Projects": ["projects", "personal projects", "key projects", "selected projects"],
    "Certifications": [
        "certifications", "certificates", "licenses and certifications",
        "certifications and licenses",
    ],
    "Publications": ["publications", "research", "papers"],
    "Awards": ["awards", "honors", "awards and honors", "achievements"],
    "Languages": ["languages"],
}
_ALIAS_TO_SECTION = {alias: name for name, aliases in SECTION_ALIASES.items() for alias in aliases}
# Aliases too common as in-entry labels ("Tech stack: Django") to start a section inline.
_NO_INLINE_ALIASES = {"tech stack", "technologies", "languages", "about", "research", "employment"}

# Sections made of independent entries (one job, one project) that a chunk should keep whole.
ENTRY_SECTIONS = {"Experience", "Projects"}

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+|\d{{1,2}}/)?(?:19|20)\d{{2}}"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE})\s*(?:-|–|—|to)\s*(?P<end>{_DATE}|present|current|now)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-•*▪◦·]\s*")
# "Skills: Python, SQL" - a header with its content on the same line.
_INLINE_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z &/]{1,40}?)\s*:\s*(\S.*)$")


# --- loading -------------------------------------------------------------------


def load_resumes(directory: str) -> list[dict]:
    """Read every supported resume in ``directory`` with the Milestone 1 tools.

    Returns ``[{"path", "file_name", "content"}]``. Unreadable, unsupported or
    empty files are skipped with a logged warning rather than aborting the run.
    """
    resumes = []
    for entry in fs_tools.list_files(directory):
        if entry["extension"] not in fs_tools.SUPPORTED_READ_EXTENSIONS:
            continue
        result = fs_tools.read_file(entry["path"])
        if not result["success"]:
            logger.warning("Skipping %s: %s", entry["name"], result["error"])
            continue
        if not result["content"].strip():
            logger.warning("Skipping %s: no extractable text", entry["name"])
            continue
        resumes.append(
            {"path": result["filepath"], "file_name": entry["name"], "content": result["content"]}
        )
    return resumes


# --- section splitting -----------------------------------------------------------


def _normalize_header(line: str) -> str:
    text = line.strip().strip("#*=_-:|•").strip().lower()
    text = text.replace("&", "and")
    return re.sub(r"\s+", " ", text)


def match_section_header(line: str) -> tuple[str, str] | None:
    """Return ``(section, inline_content)`` if ``line`` is a section header.

    ``inline_content`` is the text after a colon for one-line sections such as
    ``"Skills: Python, SQL"``, otherwise ``""``.
    """
    if not line.strip() or len(line.strip()) > 60:
        return None
    section = _ALIAS_TO_SECTION.get(_normalize_header(line))
    if section:
        return section, ""
    inline = _INLINE_HEADER_RE.match(line)
    if inline:
        alias = _normalize_header(inline.group(1))
        section = _ALIAS_TO_SECTION.get(alias)
        if section and alias not in _NO_INLINE_ALIASES:
            return section, inline.group(2).strip()
    return None


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into ``{section: text}`` in document order.

    Text before the first recognised header is stored under ``"Header"``
    (usually name and contact line). A section that appears twice is merged.
    If no header is recognised the whole text is returned as ``"General"``.
    """
    sections: dict[str, list[str]] = {HEADER_SECTION: []}
    current = HEADER_SECTION
    found_header = False
    for line in text.splitlines():
        header = match_section_header(line)
        if header:
            current, inline = header
            found_header = True
            sections.setdefault(current, [])
            if inline:
                sections[current].append(inline)
            continue
        sections[current].append(line)

    if not found_header:
        body = text.strip()
        return {GENERAL_SECTION: body} if body else {}

    result = {}
    for name, lines in sections.items():
        body = "\n".join(lines).strip()
        if body:
            result[name] = body
    return result


# --- chunking ----------------------------------------------------------------------


def _is_entry_start(line: str) -> bool:
    """A job/project title line: not a bullet and carries a date range."""
    return bool(line.strip()) and not _BULLET_RE.match(line) and bool(DATE_RANGE_RE.search(line))


def _split_entries(text: str) -> list[str]:
    """Group a section's lines into entries, each starting at a dated title line."""
    entries: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if _is_entry_start(line) or not entries:
            entries.append([line])
        else:
            entries[-1].append(line)
    return ["\n".join(e) for e in entries]


def _split_by_lines(text: str, limit: int) -> list[str]:
    """Pack lines into pieces of at most ``limit`` chars; hard-wrap a single overlong line."""
    pieces, current = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            cut = line.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(line[:cut])
            line = line[cut:].lstrip()
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        pieces.append(current)
    return pieces


def _pack(blocks: list[str], limit: int) -> list[str]:
    """Greedily join blocks into chunks of at most ``limit`` chars.

    A block is only broken apart when it alone exceeds the limit.
    """
    chunks, current = [], ""
    for block in blocks:
        if len(block) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_by_lines(block, limit))
            continue
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) > limit:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _section_pieces(section: str, body: str, limit: int) -> list[str]:
    if len(body) <= limit:
        return [body]
    blocks = _split_entries(body) if section in ENTRY_SECTIONS else body.splitlines()
    return _pack([b for b in blocks if b.strip()], limit)


def chunk_resume(
    text: str,
    candidate_name: str,
    resume_path: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """Split a resume into section-aware chunks.

    Each section becomes one chunk; a section longer than ``max_chars`` is
    split between job/project entries (Experience, Projects) or lines (others).
    A short header (name + contact) is dropped because it carries no
    matchable content; names live in metadata instead.

    Returns ``[{"id", "section", "chunk_index", "text", "document",
    "candidate_name", "resume_path"}]``, where ``text`` is the raw section
    text (used for excerpts) and ``document`` adds a
    ``"Candidate: ... | Section: ..."`` prefix that gives the embedding context.
    """
    file_name = Path(resume_path).name
    chunks = []
    index = 0
    for section, body in split_sections(text).items():
        if section == HEADER_SECTION and len(body.splitlines()) <= MAX_HEADER_LINES_TO_SKIP:
            continue
        for part, piece in enumerate(_section_pieces(section, body, max_chars)):
            chunks.append(
                {
                    "id": f"{file_name}::{section.lower()}::{part}",
                    "section": section,
                    "chunk_index": index,
                    "text": piece,
                    "document": f"Candidate: {candidate_name} | Section: {section}\n{piece}",
                    "candidate_name": candidate_name,
                    "resume_path": resume_path,
                }
            )
            index += 1
    return chunks


# --- metadata extraction --------------------------------------------------------------

DEGREE_LEVELS = {"Associate": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
_DEGREE_PATTERNS = [
    ("PhD", re.compile(r"\b(?:ph\.?\s?d\b|doctor(?:ate)?\b|d\.\s?phil\b)", re.IGNORECASE)),
    (
        "Master's",
        re.compile(
            r"\b(?:master'?s?\b|m\.\s?s\.?|m\.?\s?sc\b|ms\b|m\.\s?a\.|mba\b|m\.?\s?tech\b"
            r"|m\.\s?eng\b|meng\b|m\.\s?e\.)",
            re.IGNORECASE,
        ),
    ),
    (
        "Bachelor's",
        re.compile(
            r"\b(?:bachelor'?s?\b|b\.\s?s\.?|b\.?\s?sc\b|bs\b|b\.\s?a\.|ba\b|b\.?\s?tech\b"
            r"|b\.\s?eng\b|beng\b|b\.\s?e\.|bba\b)",
            re.IGNORECASE,
        ),
    ),
    ("Associate", re.compile(r"\bassociate(?:'s)?\s+(?:degree|of)\b", re.IGNORECASE)),
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CONTACT_LINE_RE = re.compile(r"@|https?://|www\.|linkedin|github|\d{3}", re.IGNORECASE)
_NOT_A_NAME = {"resume", "curriculum vitae", "cv", "profile"}
_NAME_PARTICLES = {"de", "del", "la", "le", "da", "di", "van", "von", "der", "den", "bin", "al"}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}
_DATE_PARTS_RE = re.compile(r"(?:(?P<mon>[a-z]+)\.?\s+|(?P<num>\d{1,2})/)?(?P<year>\d{4})")
_TOTAL_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?\bexperience", re.IGNORECASE
)
# "9 years of Python experience", "7 years of experience in Python".
_YEARS_OF_SKILL_RE = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b(?:\s+of)?"
    r"(?:\s+(?:professional|hands-on|industry|commercial))?(?:\s+experience)?"
    r"(?:\s+(?:in|with|using))?\s+(?P<rest>[^.;\n]{1,80})",
    re.IGNORECASE,
)
# A skill named in a years claim must appear right after it to be credited.
_YEARS_CLAIM_WINDOW = 25
_LIST_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z /&-]{1,30}:\s*")
MAX_RAW_SKILL_CHARS = 40


def _is_name_word(word: str) -> bool:
    letters = word.replace(".", "").replace("'", "").replace("-", "")
    return letters.isalpha() and (word[0].isupper() or word in _NAME_PARTICLES)


def _extract_name(sections: dict[str, str]) -> str | None:
    source = sections.get(HEADER_SECTION) or sections.get(GENERAL_SECTION, "")
    for line in source.splitlines()[:5]:
        line = line.strip()
        if not line or _CONTACT_LINE_RE.search(line) or line.lower() in _NOT_A_NAME:
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(_is_name_word(w) for w in words):
            return line
    return None


def _name_from_path(resume_path: str) -> str | None:
    words = [w for w in re.split(r"[_\-\s]+", Path(resume_path).stem) if w]
    words = [w for w in words if w.lower() not in {"resume", "cv"}]
    return " ".join(w.capitalize() for w in words) or None


def _split_list_items(text: str) -> list[str]:
    """Split a skills list on , ; | • and newlines, but not inside parentheses."""
    items, current, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and ch in ",;|•\n":
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    items.append("".join(current))
    return [i.strip() for i in items if i.strip()]


def _skills_from_list(text: str) -> list[str]:
    """Normalize the items of a Skills section; unknown items are kept as written."""
    skills = []
    for item in _split_list_items(text):
        item = _LIST_LABEL_RE.sub("", _BULLET_RE.sub("", item)).strip().rstrip(".")
        parts = [item]
        paren = re.match(r"^(?P<main>[^(]+)\((?P<inner>[^)]*)\)", item)
        if paren:  # "AWS (EC2, S3, Lambda)" -> AWS, EC2, S3, Lambda
            parts = [paren["main"].strip(), *re.split(r"[,;/]", paren["inner"])]
        for part in (p.strip() for p in parts):
            if not part:
                continue
            canonical = skills_vocab.normalize_skill(part)
            if canonical:
                skills.append(canonical)
            elif found := skills_vocab.find_skills(part):
                skills.extend(found)
            elif len(part) <= MAX_RAW_SKILL_CHARS:
                skills.append(part)
    return skills


def _dedupe(items: list[str]) -> list[str]:
    seen, result = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def _extract_skills(sections: dict[str, str]) -> list[str]:
    listed = _skills_from_list(sections.get("Skills", ""))
    prose = "\n".join(body for name, body in sections.items() if name != "Skills")
    return skills_vocab.with_implied(_dedupe(listed + skills_vocab.find_skills(prose)))


def _month_index(token: str, today: date, is_end: bool) -> int | None:
    """Months since year 0. End dates are exclusive (the month after the last one worked)."""
    token = token.strip().lower()
    if token in {"present", "current", "now"}:
        return today.year * 12 + today.month
    parts = _DATE_PARTS_RE.search(token)
    if not parts:
        return None
    year = int(parts["year"])
    month = _MONTHS.get((parts["mon"] or "")[:3]) or (int(parts["num"]) if parts["num"] else None)
    if month is None or not 1 <= month <= 12:
        return year * 12  # year only: "2018 - 2021" counts as 3 years
    return year * 12 + month - 1 + (1 if is_end else 0)


def _role_intervals(experience: str, today: date) -> list[tuple[int, int, str]]:
    """``(start, end, entry_text)`` per dated role, in months; undated entries are ignored."""
    roles = []
    now = today.year * 12 + today.month
    for entry in _split_entries(experience):
        dates = DATE_RANGE_RE.search(entry.splitlines()[0])
        if not dates:
            continue
        start = _month_index(dates["start"], today, is_end=False)
        end = _month_index(dates["end"], today, is_end=True)
        if start is None or end is None:
            continue
        end = min(end, now)
        if end > start:
            roles.append((start, end, entry))
    return roles


def _merged_years(intervals: list[tuple[int, int]]) -> float:
    """Total years covered by month intervals, counting overlaps once."""
    total, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return round(total / 12, 1)


def _extract_experience(
    sections: dict[str, str], today: date
) -> tuple[float | None, dict[str, float]]:
    """Return ``(total_years, {skill: years})``.

    Total years come from merged role date ranges, or failing that from a
    "N years of experience" phrase. A skill earns the years of every role whose
    text mentions it, raised to any explicit claim like "9 years of Python".
    """
    experience = sections.get("Experience") or sections.get(GENERAL_SECTION, "")
    roles = _role_intervals(experience, today)

    total = _merged_years([(s, e) for s, e, _ in roles]) if roles else None
    if total is None:
        claim = _TOTAL_YEARS_RE.search("\n".join(sections.values()))
        total = float(claim.group(1)) if claim else None

    per_skill: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, entry in roles:
        for skill in skills_vocab.with_implied(skills_vocab.find_skills(entry)):
            per_skill[skill].append((start, end))
    skill_years = {skill: _merged_years(spans) for skill, spans in per_skill.items()}

    for claim in _YEARS_OF_SKILL_RE.finditer("\n".join(sections.values())):
        named = skills_vocab.find_skills(claim["rest"][:_YEARS_CLAIM_WINDOW])
        if named:
            skill_years[named[0]] = max(skill_years.get(named[0], 0.0), float(claim["years"]))
    return total, skill_years


def degrees_mentioned(text: str) -> list[str]:
    """Every degree named in ``text``, highest first (keys of ``DEGREE_LEVELS``)."""
    return [degree for degree, pattern in _DEGREE_PATTERNS if pattern.search(text)]


def degree_level(text: str) -> str | None:
    """Highest degree named in ``text``: ``"PhD"``, ``"Master's"``, ``"Bachelor's"``, ``"Associate"``."""
    found = degrees_mentioned(text)
    return found[0] if found else None


def _extract_education(sections: dict[str, str]) -> tuple[list[str], str | None]:
    lines = [_BULLET_RE.sub("", l).strip() for l in sections.get("Education", "").splitlines()]
    lines = _dedupe([l for l in lines if l])
    return lines, degree_level("\n".join(lines))


# --- LLM fallback -------------------------------------------------------------------------

_LLM_FIELD_SCHEMAS = {
    "name": {"type": ["string", "null"], "description": "Candidate's full name."},
    "experience_years": {
        "type": ["number", "null"],
        "description": "Total years of professional work experience.",
    },
    "education": {
        "type": "array",
        "items": {"type": "string"},
        "description": "One entry per degree, e.g. 'B.S. Computer Science, MIT, 2018'.",
    },
    "highest_degree": {"type": ["string", "null"], "enum": [*DEGREE_LEVELS, None]},
}


def make_llm_client() -> openai.OpenAI | None:
    """An OpenAI client if ``OPENAI_API_KEY`` is configured, else ``None`` (no fallback)."""
    load_dotenv()
    return openai.OpenAI() if os.getenv("OPENAI_API_KEY") else None


def _llm_extract(text: str, fields: list[str], client, model: str) -> dict:
    properties = {f: _LLM_FIELD_SCHEMAS[f] for f in fields}
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Extract the requested fields from the resume. "
                "Use null (or an empty list) when the resume does not state a value.",
            },
            {"role": "user", "content": text[:LLM_MAX_INPUT_CHARS]},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "resume_metadata", "strict": True, "schema": schema},
        },
    )
    return json.loads(response.choices[0].message.content)


def _apply_llm_values(metadata: dict, values: dict) -> list[str]:
    """Copy valid LLM values into ``metadata`` and return the names of the fields filled."""
    filled = []
    name = values.get("name")
    if "name" in values and isinstance(name, str) and name.strip():
        metadata["name"] = name.strip()
        filled.append("name")
    years = values.get("experience_years")
    if isinstance(years, (int, float)) and 0 <= years <= 60:
        metadata["experience_years"] = round(float(years), 1)
        filled.append("experience_years")
    education = values.get("education")
    if isinstance(education, list) and any(isinstance(e, str) and e.strip() for e in education):
        metadata["education"] = [e.strip() for e in education if isinstance(e, str) and e.strip()]
        filled.append("education")
        degree = values.get("highest_degree")
        if degree not in DEGREE_LEVELS:
            degree = degree_level("\n".join(metadata["education"]))
        metadata["highest_degree"] = degree
    return filled


# --- public entry point --------------------------------------------------------------------


def extract_metadata(
    text: str,
    resume_path: str = "",
    llm_client=None,
    llm_model: str | None = None,
    today: date | None = None,
) -> dict:
    """Extract searchable metadata from resume text.

    Fields are parsed with rules first. If ``name``, ``experience_years`` or
    ``education`` is still missing and ``llm_client`` is given, one OpenAI
    structured-output call fills only the missing fields. A name that is still
    missing falls back to the file name.

    Returns ``{"name", "email", "skills", "experience_years", "skill_years",
    "education", "highest_degree", "llm_fields"}`` where ``llm_fields`` lists
    the fields the LLM supplied.
    """
    today = today or date.today()
    sections = split_sections(text)
    experience_years, skill_years = _extract_experience(sections, today)
    education, highest_degree = _extract_education(sections)
    email = _EMAIL_RE.search(text)
    metadata = {
        "name": _extract_name(sections),
        "email": email.group(0) if email else None,
        "skills": _extract_skills(sections),
        "experience_years": experience_years,
        "skill_years": skill_years,
        "education": education,
        "highest_degree": highest_degree,
        "llm_fields": [],
    }

    missing = [f for f in ("name", "experience_years", "education") if not metadata[f]]
    if metadata["experience_years"] == 0:
        missing.remove("experience_years")  # parsed, just no experience yet
    if missing and llm_client is not None:
        fields = missing + (["highest_degree"] if "education" in missing else [])
        model = llm_model or os.getenv("OPENAI_MODEL") or DEFAULT_LLM_MODEL
        try:
            values = _llm_extract(text, fields, llm_client, model)
            metadata["llm_fields"] = _apply_llm_values(metadata, values)
        except (openai.OpenAIError, json.JSONDecodeError, AttributeError, IndexError) as exc:
            logger.warning("LLM metadata fallback failed for %s: %s", resume_path or "resume", exc)

    if not metadata["name"]:
        metadata["name"] = _name_from_path(resume_path) if resume_path else None
    return metadata


# --- embeddings -------------------------------------------------------------------------------


class Embedder(Protocol):
    name: str  # recorded on the collection so queries use the embedder that built it

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI embeddings (``text-embedding-3-small`` by default), sent in batches."""

    def __init__(self, model: str | None = None, client=None, batch_size: int = EMBED_BATCH_SIZE):
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.client = client or openai.OpenAI()
        self.batch_size = batch_size
        self.name = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model, input=texts[i : i + self.batch_size]
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors


STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it of on or our over the to "
    "using we will with you your experience years year building built strong".split()
)


class HashingEmbedder:
    """Offline bag-of-words embedder: hashed token counts, L2-normalized.

    It only captures word overlap, not meaning. It exists for tests and for
    trying the pipeline without an API key. Stopwords (and filler such as
    "experience") are dropped so shared boilerplate does not dominate.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.name = f"hashing:{dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9+#]+", text.lower()):
            if token in STOPWORDS:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def embedder_from_name(name: str) -> Embedder:
    """Rebuild an embedder from the ``name`` stored on a collection."""
    kind, _, arg = name.partition(":")
    if kind == "openai":
        return OpenAIEmbedder(model=arg or None)
    if kind == "hashing":
        return HashingEmbedder(int(arg) if arg else 512)
    raise ValueError(f"Unknown embedder '{name}'")


def default_embedder_name() -> str:
    return f"openai:{os.getenv('OPENAI_EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL}"


# --- vector store ---------------------------------------------------------------------------------


class EmbedderMismatchError(ValueError):
    """The index was built with a different embedder than the one requested."""


def _content_hash(content: str) -> str:
    return hashlib.sha256(f"{PIPELINE_VERSION}\n{content}".encode("utf-8")).hexdigest()


def _chunk_record(chunk: dict, meta: dict, file_name: str, content_hash: str) -> dict:
    """Chroma metadata for one chunk: chunk fields plus the resume's metadata.

    Chroma accepts scalars and lists but not dicts or ``None``, so
    ``skill_years`` is stored as JSON and empty values are left out.
    """
    record = {
        "candidate_name": chunk["candidate_name"],
        "resume_path": chunk["resume_path"],
        "file_name": file_name,
        "section": chunk["section"],
        "chunk_index": chunk["chunk_index"],
        "content_hash": content_hash,
        "email": meta["email"],
        "skills": meta["skills"],
        "skill_years": json.dumps(meta["skill_years"]),
        "experience_years": meta["experience_years"],
        "education": meta["education"],
        "highest_degree": meta["highest_degree"],
        "degree_level": DEGREE_LEVELS.get(meta["highest_degree"], 0),
        "llm_fields": meta["llm_fields"],
    }
    return {key: value for key, value in record.items() if value not in (None, [], "")}


def parse_record(record: dict) -> dict:
    """Resume-level metadata from a stored chunk record (inverse of ``_chunk_record``)."""
    return {
        "candidate_name": record["candidate_name"],
        "resume_path": record["resume_path"],
        "file_name": record.get("file_name", Path(record["resume_path"]).name),
        "email": record.get("email"),
        "skills": list(record.get("skills", [])),
        "skill_years": json.loads(record.get("skill_years", "{}")),
        "experience_years": record.get("experience_years"),
        "education": list(record.get("education", [])),
        "highest_degree": record.get("highest_degree"),
        "degree_level": record.get("degree_level", 0),
    }


class ResumeRAG:
    """Indexes resumes into a persistent ChromaDB collection (cosine similarity).

    The collection records which embedder built it; opening it with a
    different one raises ``EmbedderMismatchError`` (use ``reset=True`` to
    rebuild). With no ``embedder`` given, the recorded one is reused, so the
    job matcher always embeds queries the same way the resumes were embedded.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_DB_DIR,
        embedder: Embedder | None = None,
        llm_client=None,
        collection_name: str = COLLECTION_NAME,
        reset: bool = False,
    ):
        self.persist_dir = persist_dir
        self.llm_client = llm_client
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        if reset:
            try:
                self._client.delete_collection(collection_name)
            except NotFoundError:
                pass
        self.collection = self._open_collection(collection_name)

    def _open_collection(self, name: str):
        try:
            collection = self._client.get_collection(name, embedding_function=None)
        except NotFoundError:
            embedder_name = self._embedder.name if self._embedder else default_embedder_name()
            return self._client.create_collection(
                name,
                metadata={"hnsw:space": "cosine", "embedder": embedder_name},
                embedding_function=None,
            )
        stored = (collection.metadata or {}).get("embedder")
        if self._embedder and stored and stored != self._embedder.name:
            raise EmbedderMismatchError(
                f"Index at '{self.persist_dir}' was built with '{stored}', not "
                f"'{self._embedder.name}'. Re-index with --reset to switch embedders."
            )
        return collection

    @property
    def embedder_name(self) -> str:
        return (self.collection.metadata or {}).get("embedder", default_embedder_name())

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = embedder_from_name(self.embedder_name)
        return self._embedder

    # --- indexing ---

    def index_directory(self, directory: str, force: bool = False) -> dict:
        """Index every resume in ``directory``; skip unchanged ones unless ``force``.

        Resumes previously indexed from this directory whose files are gone
        (or no longer readable) are removed from the index.

        Returns ``{"indexed", "unchanged", "removed"}`` (lists of resume paths)
        and ``"chunks"`` (number of chunks written).
        """
        summary = {"indexed": [], "unchanged": [], "removed": [], "chunks": 0}
        seen = set()
        for resume in load_resumes(directory):
            path = resume["path"]
            seen.add(path)
            content_hash = _content_hash(resume["content"])
            if not force and self._stored_hash(path) == content_hash:
                summary["unchanged"].append(path)
                continue
            summary["chunks"] += self.index_resume(resume, content_hash)
            summary["indexed"].append(path)

        directory_path = Path(directory).expanduser().resolve()
        for path in sorted(self.indexed_paths() - seen):
            if Path(path).parent == directory_path:
                self.remove_resume(path)
                summary["removed"].append(path)
        return summary

    def index_resume(self, resume: dict, content_hash: str | None = None) -> int:
        """Chunk, extract metadata, embed and store one resume from ``load_resumes``.

        Replaces any chunks already stored for the same path. Returns the
        number of chunks written.
        """
        path, content = resume["path"], resume["content"]
        meta = extract_metadata(content, path, llm_client=self.llm_client)
        chunks = chunk_resume(content, meta["name"], path)
        if not chunks:
            logger.warning("No content to index in %s", path)
            return 0
        # Embed before deleting so a failed API call leaves the old version intact.
        vectors = self.embedder.embed([c["document"] for c in chunks])
        file_name = resume.get("file_name") or Path(path).name
        content_hash = content_hash or _content_hash(content)
        path_key = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]

        self.remove_resume(path)
        self.collection.add(
            ids=[f"{c['id']}@{path_key}" for c in chunks],
            embeddings=vectors,
            documents=[c["document"] for c in chunks],
            metadatas=[_chunk_record(c, meta, file_name, content_hash) for c in chunks],
        )
        return len(chunks)

    def remove_resume(self, resume_path: str) -> None:
        self.collection.delete(where={"resume_path": resume_path})

    def _stored_hash(self, resume_path: str) -> str | None:
        found = self.collection.get(
            where={"resume_path": resume_path}, limit=1, include=["metadatas"]
        )
        return found["metadatas"][0].get("content_hash") if found["ids"] else None

    # --- reading ---

    def _all_records(self) -> list[dict]:
        return self.collection.get(include=["metadatas"])["metadatas"]

    def all_chunks(self) -> list[dict]:
        """Every stored chunk as ``{"id", "document", "metadata"}`` (for keyword search)."""
        stored = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": chunk_id, "document": document, "metadata": record}
            for chunk_id, document, record in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        ]

    def indexed_paths(self) -> set[str]:
        return {record["resume_path"] for record in self._all_records()}

    def list_resumes(self) -> list[dict]:
        """One metadata dict per indexed resume, sorted by candidate name."""
        resumes = {}
        for record in self._all_records():
            resumes.setdefault(record["resume_path"], parse_record(record))
        return sorted(resumes.values(), key=lambda r: r["candidate_name"].lower())

    def stats(self) -> dict:
        return {
            "persist_dir": self.persist_dir,
            "collection": self.collection.name,
            "embedder": self.embedder_name,
            "resumes": len(self.indexed_paths()),
            "chunks": self.collection.count(),
        }


# --- CLI --------------------------------------------------------------------------------------------


def _display_path(path: str) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return path


def _print_resume_line(meta: dict) -> None:
    years = meta["experience_years"]
    years_text = f"{years:g} yrs" if years is not None else "? yrs"
    print(
        f"  {meta['candidate_name']:<20} {years_text:>8}  {meta['highest_degree'] or '-':<11} "
        f"{len(meta['skills']):>2} skills  {_display_path(meta['resume_path'])}"
    )


def _run_index(args: argparse.Namespace) -> int:
    if args.embedder == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Add it to .env, or use --embedder hashing "
            "to index offline.",
            file=sys.stderr,
        )
        return 1
    embedder = OpenAIEmbedder() if args.embedder == "openai" else HashingEmbedder()
    llm_client = None if args.no_llm else make_llm_client()
    rag = ResumeRAG(args.db, embedder=embedder, llm_client=llm_client, reset=args.reset)
    summary = rag.index_directory(args.directory, force=args.force)

    print(
        f"Indexed {len(summary['indexed'])} resume(s) ({summary['chunks']} chunks), "
        f"{len(summary['unchanged'])} unchanged, {len(summary['removed'])} removed "
        f"[{rag.embedder_name}, db: {args.db}]"
    )
    indexed = set(summary["indexed"])
    for meta in rag.list_resumes():
        if meta["resume_path"] in indexed:
            _print_resume_line(meta)
    for path in summary["removed"]:
        print(f"  removed {_display_path(path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=DEFAULT_DB_DIR, help=f"ChromaDB directory (default: {DEFAULT_DB_DIR}).")
    parser = argparse.ArgumentParser(description="Index resumes into a ChromaDB vector store.")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", parents=[common], help="Index a directory of resumes.")
    index.add_argument("directory", nargs="?", default="resumes", help="Default: resumes")
    index.add_argument(
        "--embedder", choices=["openai", "hashing"], default="openai",
        help="openai (default) or hashing (offline, keyword-only).",
    )
    index.add_argument("--no-llm", action="store_true", help="Disable the LLM metadata fallback.")
    index.add_argument("--force", action="store_true", help="Re-index unchanged resumes too.")
    index.add_argument("--reset", action="store_true", help="Delete the index first (needed to switch embedders).")
    commands.add_parser("stats", parents=[common], help="Show index statistics.")
    commands.add_parser("list", parents=[common], help="List indexed candidates.")
    args = parser.parse_args(argv)

    try:
        if args.command == "index":
            return _run_index(args)
        rag = ResumeRAG(args.db)
        if args.command == "stats":
            for key, value in rag.stats().items():
                print(f"{key:>12}: {value}")
        else:
            resumes = rag.list_resumes()
            print(f"{len(resumes)} indexed resume(s):")
            for meta in resumes:
                _print_resume_line(meta)
        return 0
    except (FileNotFoundError, NotADirectoryError, EmbedderMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
    except openai.AuthenticationError:
        print("Error: authentication failed. Check OPENAI_API_KEY.", file=sys.stderr)
    except openai.RateLimitError as exc:
        print(f"Error: rate limited or out of quota: {exc}", file=sys.stderr)
    except openai.APIConnectionError:
        print("Error: could not reach the OpenAI API. Check your connection.", file=sys.stderr)
    except openai.OpenAIError as exc:
        print(f"Error: OpenAI API error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
