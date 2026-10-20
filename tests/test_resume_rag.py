import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import openai
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
    assert chunks[2]["id"] == "ada.txt::experience::0"
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
    assert [c["id"] for c in chunks] == [f"x.txt::experience::{i}" for i in range(len(chunks))]
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


# --- metadata extraction ------------------------------------------------------------------

TODAY = date(2026, 10, 20)


def _meta(text, **kwargs):
    return resume_rag.extract_metadata(text, today=TODAY, **kwargs)


def test_extract_metadata_from_sample_text():
    meta = _meta(SAMPLE)
    assert meta["name"] == "Ada Lovelace"
    assert meta["email"] == "ada@example.com"
    assert meta["skills"][:3] == ["Python", "Django", "PostgreSQL"]
    # Jan 2021 - Oct 2026 (70 months) + 2018 - 2020 (24 months)
    assert meta["experience_years"] == 7.8
    assert meta["education"] == ["B.S. Computer Science, MIT, 2018"]
    assert meta["highest_degree"] == "Bachelor's"
    assert meta["llm_fields"] == []


def test_name_skips_title_and_contact_lines():
    text = "RESUME\njane@x.com | 555-123-4567\nJane Q. O'Neil\nSKILLS\nPython"
    assert _meta(text)["name"] == "Jane Q. O'Neil"


def test_name_allows_particles_and_rejects_lowercase_prose():
    assert _meta("Juan de la Cruz\nSKILLS\nPython")["name"] == "Juan de la Cruz"
    assert _meta("José García\nSKILLS\nPython")["name"] == "José García"
    assert _meta("built many apps\nSKILLS\nPython")["name"] is None


def test_name_falls_back_to_file_name():
    meta = _meta("SKILLS\nPython", resume_path="resumes/resume_mary_jones.pdf")
    assert meta["name"] == "Mary Jones"


def test_skills_list_parsing():
    text = (
        "SKILLS\nLanguages: Python 3.11, Java 17; Scala\n"
        "AWS (EC2, S3, Lambda) | sklearn • Hugging Face Transformers\nWCAG accessibility"
    )
    assert _meta(text)["skills"] == [
        "Python", "Java", "Scala", "AWS", "EC2", "S3", "Lambda", "scikit-learn", "Hugging Face",
        "WCAG accessibility",
    ]


def test_skills_include_prose_mentions_and_implied_languages():
    text = "SKILLS\nDjango\nEXPERIENCE\nDev, X (2020 - 2022)\n- Shipped Kafka consumers."
    assert _meta(text)["skills"] == ["Django", "Kafka", "Python"]


def test_experience_merges_overlapping_roles():
    text = (
        "EXPERIENCE\nA, X (2015 - 2020)\n- did things\nB, Y (2018 - 2019)\n"
        "C, Z (Mar 2021 - Feb 2022)\nEDUCATION\nB.S. Math (2011 - 2015)"
    )
    # 2015-2020 (60) + Mar 2021-Feb 2022 inclusive (12) = 72 months; education dates ignored
    assert _meta(text)["experience_years"] == 6.0


def test_experience_present_is_capped_at_today_and_future_ignored():
    text = "EXPERIENCE\nA, X (Oct 2025 - Present)\nB, Y (2030 - 2031)"
    assert _meta(text)["experience_years"] == 1.1


def test_experience_falls_back_to_years_phrase():
    text = "SUMMARY\nAnalyst with 4+ years of hands-on experience.\nEXPERIENCE\nAnalyst at X"
    assert _meta(text)["experience_years"] == 4.0


def test_skill_years_from_roles_implied_and_explicit_claims():
    text = (
        "SUMMARY\nEngineer with 10 years of Go experience.\n"
        "EXPERIENCE\nDev, A (2020 - 2024)\n- Built Flask apps.\n"
        "Dev, B (2016 - 2020)\n- Wrote Python scripts and Go services."
    )
    years = _meta(text)["skill_years"]
    assert years["Python"] == 8.0  # 2016-2020 directly + 2020-2024 via Flask
    assert years["Flask"] == 4.0
    assert years["Go"] == 10.0  # explicit claim beats the 4 years from roles


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Ph.D. Physics, MIT", "PhD"),
        ("PhD in NLP", "PhD"),
        ("M.S. Statistics, UW", "Master's"),
        ("MSc Data Science", "Master's"),
        ("MBA, Wharton", "Master's"),
        ("Master of Engineering", "Master's"),
        ("B.Tech. Computer Science", "Bachelor's"),
        ("B.A. Design", "Bachelor's"),
        ("Bachelor of Science", "Bachelor's"),
        ("Associate degree in IT", "Associate"),
        ("Coursera Machine Learning certificate", None),
    ],
)
def test_degree_level(line, expected):
    assert resume_rag.degree_level(line) == expected


def test_highest_degree_wins():
    meta = _meta("EDUCATION\nB.S. CS, 2010\nPh.D. CS, 2016\nM.S. CS, 2012")
    assert meta["highest_degree"] == "PhD"


def test_sample_resumes_metadata():
    by_file = {
        Path(r["path"]).name: _meta(r["content"], resume_path=r["path"])
        for r in resume_rag.load_resumes(str(RESUMES_DIR))
    }
    assert len(by_file) == 12
    for meta in by_file.values():
        assert meta["name"] and meta["email"] and meta["skills"] and meta["education"]
        assert meta["experience_years"] > 0
    assert by_file["resume_priya_patel.docx"]["highest_degree"] == "PhD"
    assert by_file["resume_liam_oconnor.docx"]["name"] == "Liam O'Connor"
    assert by_file["resume_john_doe.pdf"]["skill_years"]["Python"] >= 5
    assert by_file["resume_daniel_kim.txt"]["skill_years"]["Python"] < 5


# --- LLM fallback ------------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, content=None, error=None):
        self.calls = []
        self._content = content
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=json.dumps(self._content))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_llm_not_called_when_rules_find_everything():
    llm = FakeLLM({})
    _meta(SAMPLE, llm_client=llm)
    assert llm.calls == []


def test_llm_fills_only_missing_fields():
    text = "jane@x.com\nSKILLS\nPython\nEXPERIENCE\nDev, X (2020 - 2022)"
    llm = FakeLLM(
        {
            "name": "Jane Roe",
            "education": ["MSc Physics, Oxford, 2019"],
            "highest_degree": None,
        }
    )
    meta = _meta(text, llm_client=llm, llm_model="test-model")

    assert meta["name"] == "Jane Roe"
    assert meta["education"] == ["MSc Physics, Oxford, 2019"]
    assert meta["highest_degree"] == "Master's"  # derived when the LLM gives none
    assert meta["experience_years"] == 2.0  # from rules, not asked
    assert meta["llm_fields"] == ["name", "education"]

    (call,) = llm.calls
    assert call["model"] == "test-model"
    schema = call["response_format"]["json_schema"]["schema"]
    assert sorted(schema["properties"]) == ["education", "highest_degree", "name"]


def test_llm_invalid_values_are_ignored():
    llm = FakeLLM({"name": "", "experience_years": -3, "education": []})
    meta = _meta("hello world", resume_path="cv_bob_stone.txt", llm_client=llm)
    assert meta["llm_fields"] == []
    assert meta["name"] == "Bob Stone"
    assert meta["experience_years"] is None


def test_llm_error_is_logged_not_raised(caplog):
    error = openai.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))
    meta = _meta("hello world", resume_path="x/resume_ann_lee.pdf", llm_client=FakeLLM(error=error))
    assert meta["name"] == "Ann Lee"
    assert "LLM metadata fallback failed" in caplog.text


def test_make_llm_client_without_key_returns_none(monkeypatch):
    monkeypatch.setattr(resume_rag, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resume_rag.make_llm_client() is None
