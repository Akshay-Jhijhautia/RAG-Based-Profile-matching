from pathlib import Path

import pytest

import job_matcher
import resume_rag

ROOT = Path(__file__).resolve().parent.parent
RESUMES_DIR = ROOT / "resumes"
JD_DIR = ROOT / "job_descriptions"


def _jd(name):
    return (JD_DIR / f"{name}.txt").read_text()


@pytest.fixture(scope="module")
def matcher(tmp_path_factory):
    rag = resume_rag.ResumeRAG(
        str(tmp_path_factory.mktemp("db")), embedder=resume_rag.HashingEmbedder()
    )
    rag.index_directory(str(RESUMES_DIR))
    return job_matcher.JobMatcher(rag)


# --- parse_job_description ------------------------------------------------------------


def test_parse_sample_ml_jd():
    parsed = job_matcher.parse_job_description(_jd("senior_python_ml"))
    assert {"Python", "PyTorch", "NLP", "Kubernetes", "AWS"} <= set(parsed["required_skills"])
    assert {"RAG", "Hugging Face", "MLflow"} <= set(parsed["preferred_skills"])
    assert not set(parsed["required_skills"]) & set(parsed["preferred_skills"])
    assert parsed["must_haves"] == [
        {
            "type": "skill_years",
            "skill": "Python",
            "min_years": 5.0,
            "source": "5+ years of Python experience building production systems",
        },
        {
            "type": "degree",
            "degree": "Bachelor's",  # "Bachelor's or Master's" -> the lower bar
            "source": "- Bachelor's or Master's degree in Computer Science or a related field",
        },
    ]  # "PhD" under Nice to have is not a must-have


@pytest.mark.parametrize(
    "line, expected",
    [
        ("5+ years Python", {"type": "skill_years", "skill": "Python", "min_years": 5.0}),
        ("At least 4 years of hands-on experience with Go", {"type": "skill_years", "skill": "Go", "min_years": 4.0}),
        ("3-5 years with React", {"type": "skill_years", "skill": "React", "min_years": 3.0}),
        ("7 + yrs in Java development", {"type": "skill_years", "skill": "Java", "min_years": 7.0}),
        ("6+ years of professional software experience", {"type": "total_years", "min_years": 6.0}),
        ("2 years of experience building web apps in Python", {"type": "skill_years", "skill": "Python", "min_years": 2.0}),
        (
            "4+ years of experience designing large distributed backend systems, ideally in Go",
            {"type": "total_years", "min_years": 4.0},
        ),
    ],
)
def test_years_requirements(line, expected):
    (requirement,) = job_matcher.parse_job_description(line)["must_haves"]
    requirement.pop("source")
    assert requirement == expected


def test_years_without_experience_context_is_ignored():
    assert job_matcher.parse_job_description("We have been in business for 20 years.")["must_haves"] == []


def test_degree_requirement_and_strictest_wins():
    jd = "Requirements:\n- B.S. in Computer Science\n- M.S. required for research roles"
    (degree,) = job_matcher.parse_job_description(jd)["must_haves"]
    assert degree["degree"] == "Master's"


def test_strictest_years_per_skill_kept():
    jd = "- 3+ years of Python\n- 5+ years of Python in production\n- 2+ years of Go"
    must = job_matcher.parse_job_description(jd)["must_haves"]
    assert {(m["skill"], m["min_years"]) for m in must} == {("Python", 5.0), ("Go", 2.0)}


def test_preferred_section_and_clauses_are_optional():
    jd = (
        "Requirements:\n- Solid CSS skills; Tailwind CSS is a plus\n- 5+ years of Java\n"
        "Nice to have:\nKubernetes experience\n- 8+ years of Scala\n"
        "Benefits:\n- Docker workshops"
    )
    parsed = job_matcher.parse_job_description(jd)
    assert parsed["required_skills"] == ["CSS", "Java", "Docker"]
    assert parsed["preferred_skills"] == ["Tailwind CSS", "Kubernetes", "Scala"]
    assert [m.get("skill") for m in parsed["must_haves"]] == ["Java"]


def test_jd_text_is_kept():
    assert job_matcher.parse_job_description("  Python dev  \n")["text"] == "Python dev"


# --- semantic_search ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "jd, expected_top",
    [("senior_python_ml", "Priya Patel"), ("java_backend", "Alex Lee"), ("frontend_react", "Maria Garcia")],
)
def test_semantic_search_ranks_obvious_fit_first(matcher, jd, expected_top):
    results = matcher.semantic_search(_jd(jd))
    assert results[0]["candidate_name"] == expected_top


def test_python_jd_ranks_python_resumes_above_java_and_mobile(matcher):
    names = [c["candidate_name"] for c in matcher.semantic_search(_jd("senior_python_ml"), k=12)]
    for python_person in ("Priya Patel", "Carlos Mendes"):
        for other in ("Alex Lee", "Sofia Rossi"):
            assert names.index(python_person) < names.index(other)


def test_semantic_search_result_shape(matcher):
    results = matcher.semantic_search(_jd("java_backend"), k=10)
    assert len(results) == 10
    assert len({c["resume_path"] for c in results}) == 10
    scores = [c["semantic_score"] for c in results]
    assert scores == sorted(scores, reverse=True)

    top = results[0]
    assert top["metadata"]["candidate_name"] == top["candidate_name"]
    assert "Java" in top["metadata"]["skills"]
    sims = [c["similarity"] for c in top["chunks"]]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in sims)
    assert not top["chunks"][0]["text"].startswith("Candidate:")  # prefix stripped
    assert top["chunks"][0]["section"] in {"Summary", "Skills", "Experience", "Education"}


def test_semantic_score_blends_best_and_top3(matcher):
    top = matcher.semantic_search(_jd("frontend_react"))[0]
    sims = [c["similarity"] for c in top["chunks"]]
    top3 = sims[:3]
    expected = 0.6 * sims[0] + 0.4 * sum(top3) / len(top3)
    assert top["semantic_score"] == pytest.approx(expected, abs=1e-4)


def test_k_limits_results(matcher):
    assert len(matcher.semantic_search(_jd("frontend_react"), k=3)) == 3


def test_empty_index_and_empty_jd(tmp_path):
    rag = resume_rag.ResumeRAG(str(tmp_path / "db"), embedder=resume_rag.HashingEmbedder())
    matcher = job_matcher.JobMatcher(rag)
    assert matcher.semantic_search("Python developer") == []
    with pytest.raises(ValueError):
        matcher.semantic_search("   ")
