from pathlib import Path

import pytest

import resume_rag

RESUMES_DIR = Path(__file__).resolve().parent.parent / "resumes"

SAMPLE = """Ada Lovelace
ada@example.com | (555) 000-0000

PROFESSIONAL SUMMARY
Engineer with 6 years of Python experience.

Technical Skills
Python, Django, PostgreSQL

## Work Experience
Senior Engineer, Acme (Jan 2021 – Present)
- Built Python services.
Engineer, Initech (2018 - 2020)
- Wrote Django APIs.

Education:
B.S. Computer Science, MIT, 2018
"""


# --- load_resumes ----------------------------------------------------------------


def test_load_resumes_reads_all_samples():
    resumes = resume_rag.load_resumes(str(RESUMES_DIR))
    assert len(resumes) == 12
    assert {Path(r["path"]).suffix for r in resumes} == {".pdf", ".docx", ".txt"}
    assert all(r["content"].strip() for r in resumes)


def test_load_resumes_skips_bad_and_unsupported_files(tmp_path, caplog):
    (tmp_path / "good.txt").write_text("Jane Doe\nSKILLS\nPython", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    resumes = resume_rag.load_resumes(str(tmp_path))
    assert [r["file_name"] for r in resumes] == ["good.txt"]
    assert "broken.pdf" in caplog.text
    assert "empty.txt" in caplog.text


def test_load_resumes_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resume_rag.load_resumes(str(tmp_path / "nope"))


# --- split_sections -----------------------------------------------------------------


def test_split_sections_maps_aliases_to_canonical_names():
    sections = resume_rag.split_sections(SAMPLE)
    assert list(sections) == ["Header", "Summary", "Skills", "Experience", "Education"]
    assert sections["Header"].startswith("Ada Lovelace")
    assert sections["Skills"] == "Python, Django, PostgreSQL"
    assert "Initech" in sections["Experience"]
    assert sections["Education"] == "B.S. Computer Science, MIT, 2018"


def test_split_sections_inline_header_content():
    sections = resume_rag.split_sections("Jane\nSkills: Python, SQL\nEducation: B.A. Math, 2015")
    assert sections["Skills"] == "Python, SQL"
    assert sections["Education"] == "B.A. Math, 2015"


def test_split_sections_inline_label_inside_entry_is_not_a_header():
    text = "EXPERIENCE\nEngineer, Acme (2019 - 2021)\nTech stack: Django, Redis"
    sections = resume_rag.split_sections(text)
    assert list(sections) == ["Experience"]
    assert "Tech stack: Django" in sections["Experience"]


def test_split_sections_merges_repeated_section():
    text = "SKILLS\nPython\nEXPERIENCE\nDev, X (2019 - 2020)\nSKILLS\nGo"
    assert resume_rag.split_sections(text)["Skills"] == "Python\nGo"


def test_split_sections_without_headers_is_general():
    assert resume_rag.split_sections("Just some text\nabout Python") == {
        "General": "Just some text\nabout Python"
    }
    assert resume_rag.split_sections("   ") == {}


def test_sentence_mentioning_section_word_is_not_a_header():
    text = "SUMMARY\nI have experience in Python and education technology."
    assert list(resume_rag.split_sections(text)) == ["Summary"]


@pytest.mark.parametrize("path", sorted(RESUMES_DIR.iterdir()), ids=lambda p: p.name)
def test_every_sample_resume_has_core_sections(path):
    content = resume_rag.fs_tools.read_file(str(path))["content"]
    sections = resume_rag.split_sections(content)
    assert {"Summary", "Skills", "Experience", "Education"} <= set(sections)


# --- chunk_resume ----------------------------------------------------------------------


def test_chunk_resume_one_chunk_per_section_and_skips_short_header():
    chunks = resume_rag.chunk_resume(SAMPLE, "Ada Lovelace", "resumes/ada.txt")
    assert [c["section"] for c in chunks] == ["Summary", "Skills", "Experience", "Education"]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]
    assert chunks[2]["id"] == "ada::experience::0"
    assert chunks[1]["document"] == (
        "Candidate: Ada Lovelace | Section: Skills\nPython, Django, PostgreSQL"
    )
    assert chunks[1]["text"] == "Python, Django, PostgreSQL"
    assert all(c["resume_path"] == "resumes/ada.txt" for c in chunks)


def test_chunk_resume_keeps_long_header():
    text = "Ada\nada@example.com\nLondon\nPassionate about analytical engines.\nSKILLS\nPython"
    chunks = resume_rag.chunk_resume(text, "Ada", "ada.txt")
    assert [c["section"] for c in chunks] == ["Header", "Skills"]


def test_chunk_resume_no_headers_gives_general_chunk():
    chunks = resume_rag.chunk_resume("Python developer for 5 years.", "X", "x.txt")
    assert [c["section"] for c in chunks] == ["General"]


def test_long_experience_splits_between_roles_not_inside_them():
    roles = []
    for i in range(6):
        roles.append(f"Engineer {i}, Company {i} (201{i} - 201{i + 1})")
        roles += [f"- Built feature {i}.{j} with Python and plenty of detail." for j in range(4)]
    text = "EXPERIENCE\n" + "\n".join(roles)
    chunks = resume_rag.chunk_resume(text, "X", "x.txt", max_chars=500)

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 500 for c in chunks)
    assert [c["id"] for c in chunks] == [f"x::experience::{i}" for i in range(len(chunks))]
    for chunk in chunks:
        assert chunk["text"].startswith("Engineer ")  # every chunk begins at a role title
    joined = "\n".join(c["text"] for c in chunks)
    assert joined == "\n".join(roles)  # nothing lost or duplicated


def test_oversized_single_block_is_split_by_lines():
    text = "SUMMARY\n" + " ".join(["word"] * 400)
    chunks = resume_rag.chunk_resume(text, "X", "x.txt", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 300 for c in chunks)
    assert " ".join(c["text"] for c in chunks).split() == ["word"] * 400


@pytest.mark.parametrize(
    "line",
    ["Engineer, Acme (2019 - Present)", "Dev, X (Jan 2021 – Mar 2023)", "Dev, Y, 05/2018 to 06/2020"],
)
def test_date_range_detects_role_lines(line):
    assert resume_rag._is_entry_start(line)


def test_bullet_with_dates_is_not_a_role_line():
    assert not resume_rag._is_entry_start("- Migrated systems from 2019 - 2020")
