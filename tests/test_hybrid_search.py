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


def _by_name(results):
    return {c["candidate_name"]: c for c in results}


# --- tokenizer and scoring helpers -----------------------------------------------------------


def test_tokenize_keeps_symbol_skills_and_drops_stopwords():
    assert job_matcher.tokenize("Experience with C++, C#, Node.js and CI/CD.") == [
        "c++", "c#", "node.js", "ci", "cd"
    ]


def test_hybrid_score_weights():
    assert job_matcher.hybrid_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert job_matcher.hybrid_score(0.4, 0.5, 0.5) == pytest.approx(0.5 * 0.4 + 0.2 * 0.5 + 0.3 * 0.5)
    # Without required skills the skill weight is redistributed.
    assert job_matcher.hybrid_score(1.0, 1.0, None) == pytest.approx(1.0)
    assert job_matcher.hybrid_score(0.7, 0.0, None) == pytest.approx(0.5)


# --- keyword search ----------------------------------------------------------------------------


def test_keyword_scores_are_normalized(matcher):
    scores = matcher.keyword_scores(job_matcher.parse_job_description(_jd("java_backend")))
    assert len(scores) == 12
    assert max(scores.values()) == 1.0
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    best = max(scores, key=scores.get)
    assert Path(best).name == "resume_alex_lee.txt"


def test_keyword_scores_match_aliases_via_skill_tokens(matcher):
    # "sklearn" and "ML" never appear in the resumes; their canonical skills do.
    parsed = job_matcher.parse_job_description("Staff ML engineer, sklearn")
    assert parsed["required_skills"] == ["Machine Learning", "scikit-learn"]
    scores = matcher.keyword_scores(parsed)
    names = {Path(p).name: s for p, s in scores.items()}
    assert names["resume_jane_smith.docx"] > 0
    assert names["resume_alex_lee.txt"] == 0


def test_keyword_scores_empty_index(tmp_path):
    rag = resume_rag.ResumeRAG(str(tmp_path / "db"), embedder=resume_rag.HashingEmbedder())
    parsed = job_matcher.parse_job_description("Python")
    assert job_matcher.JobMatcher(rag).keyword_scores(parsed) == {}


# --- hybrid search -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "jd, expected_top",
    [("senior_python_ml", "Priya Patel"), ("java_backend", "Alex Lee"), ("frontend_react", "Maria Garcia")],
)
def test_hybrid_search_ranks_obvious_fit_first(matcher, jd, expected_top):
    assert matcher.hybrid_search(_jd(jd))[0]["candidate_name"] == expected_top


def test_hybrid_result_fields(matcher):
    results = matcher.hybrid_search(_jd("java_backend"))
    assert len(results) == 10
    scores = [c["hybrid_score"] for c in results]
    assert scores == sorted(scores, reverse=True)

    alex = results[0]
    assert alex["matched_skills"] == ["Java", "Spring Boot", "Kafka", "Oracle", "Kubernetes"]
    assert alex["missing_skills"] == ["PostgreSQL", "CI/CD"]
    assert alex["skill_coverage"] == pytest.approx(5 / 7, abs=1e-4)
    assert alex["keyword_score"] == 1.0
    assert alex["hybrid_score"] == pytest.approx(
        job_matcher.hybrid_score(alex["semantic_score"], 1.0, alex["skill_coverage"]), abs=1e-4
    )
    skills_chunk = next(c for c in alex["chunks"] if c["section"] == "Skills")
    assert {"Java", "Spring Boot", "Kafka"} <= set(skills_chunk["skills"])


def test_preferred_skills_reported_separately(matcher):
    priya = _by_name(matcher.hybrid_search(_jd("senior_python_ml")))["Priya Patel"]
    assert {"RAG", "Hugging Face", "MLflow"} <= set(priya["matched_preferred_skills"])
    assert "RAG" not in priya["matched_skills"]


def test_jd_without_skills_has_no_coverage(matcher):
    results = matcher.hybrid_search("Someone who ships reliable software with a great team.")
    assert results
    assert all(c["skill_coverage"] is None and c["matched_skills"] == [] for c in results)


def test_keyword_hit_outside_semantic_pool_is_added(matcher, monkeypatch):
    # Shrink the semantic pool to a single chunk; Flutter is only on Sofia's resume.
    monkeypatch.setattr(job_matcher, "CHUNK_POOL", 1)
    jd = "Accounting manager for our finance team. Flutter."
    query = matcher._embed_query(jd)
    semantic_only = matcher._semantic_candidates(query)
    assert "Sofia Rossi" not in {c["candidate_name"] for c in semantic_only.values()}

    sofia = _by_name(matcher.hybrid_search(jd))["Sofia Rossi"]
    assert sofia["matched_skills"] == ["Flutter"]
    assert sofia["chunks"]  # semantic score filled from her own chunks
    assert sofia["semantic_score"] == pytest.approx(sofia["chunks"][0]["similarity"], abs=1e-4)


def test_semantic_candidates_restricted_to_paths(matcher):
    query = matcher._embed_query("Python")
    paths = [str(RESUMES_DIR / "resume_alex_lee.txt")]
    assert list(matcher._semantic_candidates(query, paths)) == paths
    assert matcher._semantic_candidates(query, []) == {}
