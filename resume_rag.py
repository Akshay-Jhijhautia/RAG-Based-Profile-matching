"""RAG pipeline for resumes: load, chunk by section, extract metadata, embed, store.

Resumes are read with the Milestone 1 file system tools (``fs_tools``), split
into their natural sections (Summary, Skills, Experience, Education, ...) and
chunked so that a chunk never mixes two sections and, where possible, never
splits a single job entry.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fs_tools

logger = logging.getLogger(__name__)

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
