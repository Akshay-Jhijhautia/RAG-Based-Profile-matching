"""RAG pipeline for resumes: load, chunk by section, extract metadata, embed, store.

Resumes are read with the Milestone 1 file system tools (``fs_tools``), split
into their natural sections (Summary, Skills, Experience, Education, ...) and
chunked so that a chunk never mixes two sections and, where possible, never
splits a single job entry.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import openai
from dotenv import load_dotenv

import fs_tools
import skills_vocab

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "gpt-4o-mini"
LLM_MAX_INPUT_CHARS = 12000

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
    stem = Path(resume_path).stem
    chunks = []
    index = 0
    for section, body in split_sections(text).items():
        if section == HEADER_SECTION and len(body.splitlines()) <= MAX_HEADER_LINES_TO_SKIP:
            continue
        for part, piece in enumerate(_section_pieces(section, body, max_chars)):
            chunks.append(
                {
                    "id": f"{stem}::{section.lower()}::{part}",
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


def degree_level(text: str) -> str | None:
    """Highest degree named in ``text``: ``"PhD"``, ``"Master's"``, ``"Bachelor's"``, ``"Associate"``."""
    for degree, pattern in _DEGREE_PATTERNS:  # ordered highest first
        if pattern.search(text):
            return degree
    return None


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
