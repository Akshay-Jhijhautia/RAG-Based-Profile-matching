import json
from pathlib import Path

import pytest

import job_matcher
import resume_rag

ROOT = Path(__file__).resolve().parent.parent
RESUMES_DIR = ROOT / "resumes"
JD_DIR = ROOT / "job_descriptions"
MATCH_KEYS = {
    "candidate_name", "resume_path", "match_score", "matched_skills", "relevant_excerpts", "reasoning"
}


def _jd(name):
    return (JD_DIR / f"{name}.txt").read_text()


@pytest.fixture(scope="module")
def db_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("db"))
    resume_rag.ResumeRAG(path, embedder=resume_rag.HashingEmbedder()).index_directory(str(RESUMES_DIR))
    return path


@pytest.fixture(scope="module")
def matcher(db_dir):
    return job_matcher.JobMatcher(resume_rag.ResumeRAG(db_dir))


def _names(result):
    return [m["candidate_name"] for m in result["top_matches"]]


# --- parse_must_have_spec --------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("Python:5", {"type": "skill_years", "skill": "Python", "min_years": 5.0}),
        ("sklearn:2+", {"type": "skill_years", "skill": "scikit-learn", "min_years": 2.0}),
        ("Kubernetes", {"type": "skill", "skill": "Kubernetes"}),
        ("Elixir", {"type": "skill", "skill": "Elixir"}),
        ("experience:6", {"type": "total_years", "min_years": 6.0}),
        ("degree:masters", {"type": "degree", "degree": "Master's"}),
        ("degree:PhD", {"type": "degree", "degree": "PhD"}),
        ("degree:bachelor", {"type": "degree", "degree": "Bachelor's"}),
    ],
)
def test_parse_must_have_spec(spec, expected):
    parsed = job_matcher.parse_must_have_spec(spec)
    assert parsed.pop("source") == spec
    assert parsed == expected


@pytest.mark.parametrize("spec", ["", ":5", "Python:five", "experience", "degree:wizard"])
def test_parse_must_have_spec_errors(spec):
    with pytest.raises(ValueError):
        job_matcher.parse_must_have_spec(spec)


# --- check_must_haves --------------------------------------------------------------------------

META = {
    "skills": ["Python", "Docker", "Go"],
    "skill_years": {"Python": 6.0, "Go": 1.5},
    "experience_years": 7.0,
    "education": ["M.S. CS"],
    "highest_degree": "Master's",
    "degree_level": 3,
}


@pytest.mark.parametrize(
    "spec, status",
    [
        ("Python:5", "met"),
        ("Go:3", "not_met"),
        ("Docker:2", "unverified"),  # listed, but no role mentions it
        ("Rust:1", "not_met"),
        ("Docker", "met"),
        ("Rust", "not_met"),
        ("experience:7", "met"),
        ("experience:8", "not_met"),
        ("degree:bachelors", "met"),
        ("degree:masters", "met"),
        ("degree:phd", "not_met"),
    ],
)
def test_check_must_haves(spec, status):
    (check,) = job_matcher.check_must_haves(META, [job_matcher.parse_must_have_spec(spec)])
    assert check["status"] == status


def test_check_must_haves_unknown_data_is_unverified():
    meta = {**META, "experience_years": None, "education": [], "highest_degree": None, "degree_level": 0}
    checks = job_matcher.check_must_haves(
        meta, [job_matcher.parse_must_have_spec(s) for s in ("experience:3", "degree:bachelors")]
    )
    assert [c["status"] for c in checks] == ["unverified", "unverified"]


def test_check_detail_text():
    (check,) = job_matcher.check_must_haves(META, [job_matcher.parse_must_have_spec("Python:5")])
    assert check == {"requirement": "5+ yrs Python", "status": "met", "detail": "6 yrs Python (needs 5+)"}


# --- excerpts and reasoning ------------------------------------------------------------------------


def test_select_excerpts_prefers_skill_chunks_and_trims():
    chunks = [
        {"section": "Summary", "text": "Generic summary", "similarity": 0.50, "skills": []},
        {"section": "Experience", "text": "Built " + "x " * 400, "similarity": 0.49, "skills": ["Python"]},
        {"section": "Skills", "text": "Python,\nGo", "similarity": 0.30, "skills": ["Python", "Go"]},
        {"section": "Education", "text": "B.S.", "similarity": 0.10, "skills": []},
    ]
    excerpts = job_matcher.select_excerpts(chunks)
    assert len(excerpts) == 3
    assert excerpts[0].startswith("[Experience] Built")
    assert excerpts[0].endswith("…") and len(excerpts[0]) <= len("[Experience] ") + 301
    assert excerpts[1] == "[Summary] Generic summary"
    assert excerpts[2] == "[Skills] Python, Go"  # newlines collapsed


def test_build_reasoning_mentions_gaps_and_unverified():
    candidate = {
        "chunks": [{"section": "Experience", "text": "", "similarity": 0.5, "skills": ["Python"]}],
        "matched_skills": ["Python"],
        "missing_skills": ["Go"],
        "matched_preferred_skills": ["Docker"],
        "skill_coverage": 0.5,
        "metadata": {"experience_years": 4.5},
    }
    checks = [
        {"requirement": "x", "status": "met", "detail": "6 yrs Python (needs 5+)"},
        {"requirement": "y", "status": "unverified", "detail": "lists Docker but years unclear"},
    ]
    text = job_matcher.build_reasoning(candidate, 60, checks)
    assert text == (
        "Good match: closest resume sections are Experience. Has 1/2 required skills (missing Go), "
        "plus preferred Docker. Experience mentions Python. Meets must-haves: 6 yrs Python (needs 5+). "
        "Unverified: lists Docker but years unclear. 4.5 years of total experience."
    )


# --- match -----------------------------------------------------------------------------------------


def test_match_output_follows_spec(matcher):
    result = matcher.match(_jd("senior_python_ml"))
    assert set(result) == {"job_description", "top_matches"}
    assert result["job_description"] == _jd("senior_python_ml").strip()
    assert result["top_matches"]
    for match in result["top_matches"]:
        assert set(match) == MATCH_KEYS
        assert isinstance(match["match_score"], int) and 0 <= match["match_score"] <= 100
        assert match["resume_path"].startswith("resumes/") or Path(match["resume_path"]).is_absolute()
        assert 1 <= len(match["relevant_excerpts"]) <= 3
        assert match["reasoning"].endswith(".")
    scores = [m["match_score"] for m in result["top_matches"]]
    assert scores == sorted(scores, reverse=True)
    assert result["top_matches"][0]["candidate_name"] == "Priya Patel"


def test_jd_must_haves_filter_candidates(matcher):
    # "5+ years of Python" in the JD removes Daniel, Liam, Olivia, Wei and non-Python resumes.
    names = _names(matcher.match(_jd("senior_python_ml")))
    assert set(names) == {"Priya Patel", "Carlos Mendes", "Jane Smith", "John Doe", "Fatima Ahmed"}


def test_ignoring_jd_must_haves_restores_pool(matcher):
    result = matcher.match(_jd("senior_python_ml"), use_jd_must_haves=False)
    assert len(result["top_matches"]) == 10


def test_extra_must_have_and_include_filtered(matcher):
    must = [job_matcher.parse_must_have_spec("degree:phd")]
    result = matcher.match(_jd("senior_python_ml"), must_haves=must, include_filtered=True)
    assert _names(result) == ["Priya Patel"]
    filtered = {f["candidate_name"]: f for f in result["filtered_out"]}
    assert len(filtered) == 11
    assert filtered["Jane Smith"]["failed_requirements"] == ["Master's (needs PhD or higher)"]
    # The JD's Bachelor's requirement is superseded by the stricter PhD one.
    assert filtered["Alex Lee"]["failed_requirements"] == [
        "no Python on resume (needs 5+ yrs Python)", "Bachelor's (needs PhD or higher)"
    ]


def test_k_counts_survivors_after_filtering(matcher):
    # Only Alex has Java, so asking for 2 still returns just the one survivor...
    must = [job_matcher.parse_must_have_spec("Java")]
    assert _names(matcher.match(_jd("java_backend"), k=2, must_haves=must)) == ["Alex Lee"]
    # ...and without that filter k is filled from the ranked pool.
    assert len(matcher.match(_jd("java_backend"), k=3)["top_matches"]) == 3


def test_details_add_breakdown(matcher):
    result = matcher.match(_jd("java_backend"), k=1, details=True)
    assert result["requirements"]["required_skills"][:2] == ["Java", "Spring Boot"]
    assert "filtered_out" in result
    details = result["top_matches"][0]["details"]
    assert details["missing_skills"] == ["PostgreSQL", "CI/CD"]
    assert {c["status"] for c in details["must_have_checks"]} == {"met"}
    expected = job_matcher.hybrid_score(
        details["semantic_score"], details["keyword_score"], details["skill_coverage"]
    )
    assert result["top_matches"][0]["match_score"] == round(100 * expected)


# --- CLI ----------------------------------------------------------------------------------------------


def test_cli_writes_json(db_dir, tmp_path, capsys):
    out = tmp_path / "results" / "match.json"
    code = job_matcher.main(
        ["--db", db_dir, "--jd-file", str(JD_DIR / "frontend_react.txt"), "--k", "2", "--out", str(out)]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text()) == printed
    assert [m["candidate_name"] for m in printed["top_matches"]][0] == "Maria Garcia"


def test_cli_inline_jd_with_must_have(db_dir, capsys):
    code = job_matcher.main(["--db", db_dir, "--jd", "Python engineer", "--must-have", "degree:phd"])
    assert code == 0
    assert _names(json.loads(capsys.readouterr().out)) == ["Priya Patel"]


@pytest.mark.parametrize(
    "args, message",
    [
        (["--jd", "x", "--must-have", "degree:wizard"], "Unknown degree"),
        (["--jd-file", "missing_jd.txt"], "Could not read job description"),
    ],
)
def test_cli_errors(db_dir, capsys, args, message):
    assert job_matcher.main(["--db", db_dir, *args]) == 1
    assert message in capsys.readouterr().err


def test_cli_empty_index(tmp_path, capsys):
    assert job_matcher.main(["--db", str(tmp_path / "db"), "--jd", "Python"]) == 1
    assert "is empty" in capsys.readouterr().err
