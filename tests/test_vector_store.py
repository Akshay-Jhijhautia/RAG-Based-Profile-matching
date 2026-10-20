import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import resume_rag

RESUMES_DIR = Path(__file__).resolve().parent.parent / "resumes"
SAMPLES = ["resume_john_doe.pdf", "resume_jane_smith.docx", "resume_alex_lee.txt"]


@pytest.fixture
def resume_dir(tmp_path):
    target = tmp_path / "resumes"
    target.mkdir()
    for name in SAMPLES:
        shutil.copy(RESUMES_DIR / name, target / name)
    return target


@pytest.fixture
def db_dir(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def rag(db_dir):
    return resume_rag.ResumeRAG(db_dir, embedder=resume_rag.HashingEmbedder())


# --- embedders ---------------------------------------------------------------------------


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = resume_rag.HashingEmbedder(dimensions=64)
    first, again = embedder.embed(["Python Django developer"] * 2)
    assert first == again
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


def test_hashing_embedder_ranks_word_overlap_higher():
    embedder = resume_rag.HashingEmbedder()
    query, python_doc, java_doc = embedder.embed(
        ["Python Django engineer", "Senior Python engineer using Django", "Java Spring Kafka"]
    )
    assert _cosine(query, python_doc) > _cosine(query, java_doc)


class FakeEmbeddingsClient:
    def __init__(self):
        self.batches = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.batches.append(list(input))
        # Return items out of order to check they are re-sorted by index.
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedder_batches_and_keeps_order():
    client = FakeEmbeddingsClient()
    embedder = resume_rag.OpenAIEmbedder(model="m", client=client, batch_size=2)
    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [len(b) for b in client.batches] == [2, 2, 1]
    assert embedder.name == "openai:m"


def test_embedder_from_name():
    assert resume_rag.embedder_from_name("hashing:128").dimensions == 128
    with pytest.raises(ValueError):
        resume_rag.embedder_from_name("word2vec")


# --- indexing ----------------------------------------------------------------------------------


def test_index_directory_stores_chunks_with_metadata(rag, resume_dir):
    summary = rag.index_directory(str(resume_dir))

    assert len(summary["indexed"]) == 3
    assert summary["unchanged"] == [] and summary["removed"] == []
    stats = rag.stats()
    assert stats["resumes"] == 3
    assert stats["chunks"] == summary["chunks"] == rag.collection.count()
    assert stats["embedder"] == "hashing:512"

    stored = rag.collection.get(include=["documents", "metadatas"])
    assert len(set(stored["ids"])) == len(stored["ids"])
    sections = {m["section"] for m in stored["metadatas"]}
    assert {"Summary", "Skills", "Experience", "Education"} <= sections
    assert all(doc.startswith("Candidate: ") for doc in stored["documents"])


def test_metadata_round_trip(rag, resume_dir):
    rag.index_directory(str(resume_dir))
    by_name = {r["candidate_name"]: r for r in rag.list_resumes()}

    assert list(by_name) == ["Alex Lee", "Jane Smith", "John Doe"]
    john = by_name["John Doe"]
    assert john["resume_path"] == str(resume_dir / "resume_john_doe.pdf")
    assert john["email"] == "john.doe@example.com"
    assert "Django" in john["skills"]
    assert john["skill_years"]["Python"] >= 5
    assert john["experience_years"] > 5
    assert john["highest_degree"] == "Bachelor's" and john["degree_level"] == 2
    assert by_name["Jane Smith"]["degree_level"] == 3


def test_reindex_skips_unchanged_and_force_does_not_duplicate(rag, resume_dir):
    first = rag.index_directory(str(resume_dir))
    count = rag.collection.count()

    again = rag.index_directory(str(resume_dir))
    assert again["indexed"] == [] and len(again["unchanged"]) == 3

    forced = rag.index_directory(str(resume_dir), force=True)
    assert len(forced["indexed"]) == 3
    assert rag.collection.count() == count == first["chunks"]


def test_changed_resume_is_replaced(rag, resume_dir):
    rag.index_directory(str(resume_dir))
    path = resume_dir / "resume_alex_lee.txt"
    path.write_text(path.read_text().replace("Kafka", "RabbitMQ"))

    summary = rag.index_directory(str(resume_dir))
    assert summary["indexed"] == [str(path)]
    docs = rag.collection.get(where={"resume_path": str(path)}, include=["documents"])["documents"]
    assert any("RabbitMQ" in d for d in docs)
    assert not any("Kafka" in d for d in docs)


def test_deleted_resume_is_removed(rag, resume_dir):
    rag.index_directory(str(resume_dir))
    (resume_dir / "resume_jane_smith.docx").unlink()

    summary = rag.index_directory(str(resume_dir))
    assert summary["removed"] == [str(resume_dir / "resume_jane_smith.docx")]
    assert rag.stats()["resumes"] == 2


def test_other_directories_are_left_alone(rag, resume_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(RESUMES_DIR / "resume_wei_chen.pdf", other)
    rag.index_directory(str(resume_dir))
    rag.index_directory(str(other))
    assert rag.stats()["resumes"] == 4


def test_failed_embedding_keeps_previous_version(rag, resume_dir):
    rag.index_directory(str(resume_dir))
    count = rag.collection.count()

    class Broken:
        name = "hashing:512"

        def embed(self, texts):
            raise RuntimeError("API down")

    rag._embedder = Broken()
    with pytest.raises(RuntimeError):
        rag.index_directory(str(resume_dir), force=True)
    assert rag.collection.count() == count


def test_index_is_persistent_and_reuses_stored_embedder(rag, resume_dir, db_dir):
    rag.index_directory(str(resume_dir))
    reopened = resume_rag.ResumeRAG(db_dir)
    assert reopened.stats()["resumes"] == 3
    assert reopened.embedder.name == "hashing:512"


def test_embedder_mismatch_requires_reset(rag, resume_dir, db_dir):
    rag.index_directory(str(resume_dir))
    with pytest.raises(resume_rag.EmbedderMismatchError):
        resume_rag.ResumeRAG(db_dir, embedder=resume_rag.HashingEmbedder(dimensions=64))

    rebuilt = resume_rag.ResumeRAG(
        db_dir, embedder=resume_rag.HashingEmbedder(dimensions=64), reset=True
    )
    assert rebuilt.collection.count() == 0
    assert rebuilt.embedder_name == "hashing:64"


def test_stored_vectors_support_similarity_query(rag, resume_dir):
    rag.index_directory(str(resume_dir))
    (query,) = rag.embedder.embed(["Java Spring Boot Kafka developer"])
    result = rag.collection.query(query_embeddings=[query], n_results=1, include=["metadatas"])
    assert result["metadatas"][0][0]["candidate_name"] == "Alex Lee"


# --- CLI -------------------------------------------------------------------------------------------


def test_cli_index_stats_and_list(resume_dir, db_dir, capsys):
    assert resume_rag.main(
        ["index", str(resume_dir), "--db", db_dir, "--embedder", "hashing", "--no-llm"]
    ) == 0
    assert "Indexed 3 resume(s)" in capsys.readouterr().out

    assert resume_rag.main(["stats", "--db", db_dir]) == 0
    assert "resumes: 3" in capsys.readouterr().out

    assert resume_rag.main(["list", "--db", db_dir]) == 0
    out = capsys.readouterr().out
    assert "3 indexed resume(s)" in out and "John Doe" in out


def test_cli_requires_api_key_for_openai(resume_dir, db_dir, monkeypatch, capsys):
    monkeypatch.setattr(resume_rag, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resume_rag.main(["index", str(resume_dir), "--db", db_dir]) == 1
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def test_cli_missing_directory(db_dir, tmp_path, capsys):
    code = resume_rag.main(
        ["index", str(tmp_path / "nope"), "--db", db_dir, "--embedder", "hashing", "--no-llm"]
    )
    assert code == 1
    assert "Directory not found" in capsys.readouterr().err
