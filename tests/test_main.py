import json
from pathlib import Path

import main

ROOT = Path(__file__).resolve().parent.parent


def test_index_then_match(tmp_path, capsys):
    db = str(tmp_path / "db")
    args = ["index", str(ROOT / "resumes"), "--db", db, "--embedder", "hashing", "--no-llm"]
    assert main.main(args) == 0
    assert "Indexed 12 resume(s)" in capsys.readouterr().out

    jd = str(ROOT / "job_descriptions" / "java_backend.txt")
    assert main.main(["match", "--db", db, "--jd-file", jd, "--k", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["top_matches"][0]["candidate_name"] == "Alex Lee"

    assert main.main(["stats", "--db", db]) == 0
    assert "resumes: 12" in capsys.readouterr().out


def test_help_and_unknown_command(capsys):
    assert main.main(["--help"]) == 0
    assert "python main.py match" in capsys.readouterr().out
    assert main.main([]) == 1
    assert main.main(["deploy"]) == 2
    assert "Unknown command" in capsys.readouterr().err
