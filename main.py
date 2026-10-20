"""Entry point for the resume RAG matcher.

    python main.py index [resumes/] [options]   build or refresh the vector index
    python main.py stats | list                 inspect the index
    python main.py match --jd-file JD [options] rank resumes for a job description

Each command forwards its options to ``resume_rag.py`` or ``job_matcher.py``;
add ``--help`` after a command to see them.
"""

from __future__ import annotations

import sys

import job_matcher
import resume_rag

INDEX_COMMANDS = {"index", "stats", "list"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0 if argv else 1
    command, rest = argv[0], argv[1:]
    if command in INDEX_COMMANDS:
        return resume_rag.main([command, *rest])
    if command == "match":
        return job_matcher.main(rest)
    print(f"Unknown command '{command}'. Use index, stats, list or match.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
