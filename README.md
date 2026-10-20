# RAG-based Resume ↔ Job Matching

Milestone 2 of the resume assistant. Resumes are chunked by section, enriched
with extracted metadata, embedded and stored in ChromaDB (`resume_rag.py`). A
job description is then matched against them with hybrid search (semantic +
BM25 keyword + critical skills), must-have filters and a 0–100 score with
written reasoning (`job_matcher.py`).

```
python resume_rag.py index resumes
python job_matcher.py --jd-file job_descriptions/senior_python_ml.txt
```

## Project layout

| Path | Purpose |
| --- | --- |
| `resume_rag.py` | Part A: loading, section chunking, metadata extraction, embeddings, ChromaDB storage, `index`/`stats`/`list` CLI |
| `job_matcher.py` | Part B: JD parsing, semantic/keyword/hybrid search, must-have filters, scoring, reasoning, JSON output CLI |
| `skills_vocab.py` | Shared skills vocabulary: aliases (`sklearn` → `scikit-learn`), free-text detection, implied languages |
| `fs_tools.py` | File system tools from Milestone 1, used to read resumes/JDs and write results |
| `main.py` | Single entry point: `python main.py index|stats|list|match ...` |
| `resumes/` | 12 fictional sample resumes (PDF, DOCX, TXT) |
| `job_descriptions/` | Sample JDs: senior Python/ML, Java backend, frontend React |
| `scripts/generate_sample_resumes.py` | Regenerates `resumes/` |
| `tests/` | pytest suite (offline: fake OpenAI clients and a hashing embedder) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

| Variable | Default | Used for |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Embeddings and the metadata fallback |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings |
| `OPENAI_MODEL` | `gpt-4o-mini` | Metadata fallback |

No key? Add `--embedder hashing` to `index` to run everything offline. That
embedder only measures word overlap, so use it for trying things out, not for
real matching.

## Usage

```bash
# Build or refresh the index (unchanged resumes are skipped)
python resume_rag.py index resumes
python resume_rag.py index resumes --force           # re-process everything
python resume_rag.py index resumes --reset --embedder hashing   # rebuild with another embedder
python resume_rag.py stats
python resume_rag.py list

# Match a job description
python job_matcher.py --jd-file job_descriptions/senior_python_ml.txt
python job_matcher.py --jd "Backend engineer, Django, 3+ years of Python" --k 5
python job_matcher.py --jd-file jd.pdf --must-have "Python:5" --must-have "degree:masters"
python job_matcher.py --jd-file jd.txt --include-filtered --details --out results/match.json
```

`main.py` accepts the same commands: `python main.py index resumes`,
`python main.py match --jd-file ...`.

| `job_matcher.py` option | Effect |
| --- | --- |
| `--jd` / `--jd-file` | JD text, or a TXT/MD/PDF/DOCX file |
| `--k` | Number of matches (default 10), counted after filtering |
| `--must-have SPEC` | Extra requirement, repeatable: `Python:5` (5+ yrs), `Python` (has skill), `experience:5`, `degree:masters` |
| `--ignore-jd-requirements` | Don't apply must-haves parsed from the JD text |
| `--include-filtered` | Add `filtered_out`: removed candidates and the requirements they failed |
| `--details` | Add parsed requirements and a per-match score breakdown |
| `--out PATH` | Also write the JSON to a file |
| `--db DIR` | ChromaDB directory (default `chroma_db`) |

## How it works

### Part A — indexing (`resume_rag.py`)

1. **Load.** `fs_tools.list_files` and `read_file` read every PDF, DOCX,
   TXT and MD file. Unreadable or empty files are skipped with a warning.
2. **Split into sections.** Headers are matched against about 50 spellings
   ("PROFESSIONAL SUMMARY", "Technical Skills", "## Work Experience",
   "Education:") in 9 canonical sections. One-line headers such as
   `Skills: Python, SQL` also count.
3. **Chunk.** Each section becomes one chunk. A section over 1,500 characters is
   split between job or project entries (lines with a date range), so a role is
   never cut in half. The name/contact header is not embedded. Each chunk is
   embedded as `Candidate: <name> | Section: <section>\n<text>`.
4. **Extract metadata.** These rules run first:
   - `name`: first header line made of 2–5 capitalised words
   - `email`: regex
   - `skills`: Skills-section items plus vocabulary skills found in the other sections, plus implied languages (Django → Python)
   - `experience_years`: job date ranges with overlapping roles counted once. "Present" means today.
   - `skill_years`: the years of every role whose text mentions the skill, raised to explicit claims such as "9 years of Python"
   - `education` and `highest_degree` (PhD > Master's > Bachelor's > Associate)

   If the name, experience or education is still missing, one OpenAI call
   with strict JSON-schema output fills only those fields. It is skipped
   without a key; `--no-llm` turns it off.
5. **Embed and store.** OpenAI `text-embedding-3-small` is called in batches.
   Chunks go into a persistent ChromaDB collection (`resumes`, cosine
   distance). Each chunk carries its section and the resume's metadata.
   `skills`/`education` are stored as lists, `skill_years` as JSON, and
   `degree_level` as a number from 0 to 4.
6. **Re-index.** A content hash skips unchanged resumes. A changed resume has
   its chunks replaced, and resumes deleted from the folder are removed. The
   collection records its embedder, so queries always use the same one;
   switching embedders needs `--reset`.

### Part B — matching (`job_matcher.py`)

1. **Parse the JD.**
   - Skills under "Nice to have", "Preferred" or "Bonus" headings, or in
     clauses saying "a plus", are preferred. All other skills are required.
   - Must-haves come from required lines only:
     - "5+ years of Python" → `skill_years`
     - "6+ years of experience" → `total_years`
     - "Bachelor's or Master's degree" → `degree` (the lower bar)
2. **Semantic search.** The JD is embedded and the 50 nearest chunks are
   retrieved. They are grouped per resume, and
   `semantic = 0.6 × best chunk + 0.4 × mean(top 3 chunks)`.
3. **Keyword search.** BM25 runs over one document per resume. Canonical skill
   tokens are added on both sides, so "ML" in a JD matches "machine learning"
   in a resume. Required skills count double. Scores are scaled so the best
   resume gets 1.0.
4. **Critical skills.** The JD's required skills are checked against each
   candidate's extracted skills: matched, missing and coverage.
5. **Hybrid pool.** The pool is the semantic hits plus the 20 best keyword hits,
   so an exact skill match is not lost when its embedding ranks low.
6. **Filter.** Each must-have is `met`, `not_met` (the candidate is removed)
   or `unverified` (the resume lists the skill but its years can't be
   determined, so the candidate is kept and flagged in the reasoning).
7. **Score and explain.** The 0–100 score is computed as below. The excerpts
   are the top 3 chunks by similarity, nudged towards chunks that mention JD
   skills. The reasoning covers fit, skill coverage, which sections mention
   which skills, the must-haves and total experience.

### Scoring

```
match_score = round(100 × (0.5 × semantic + 0.2 × keyword + 0.3 × skill_coverage))
```

- **semantic**: the cosine similarity rescaled to 0..1 over the embedder's
  typical range: OpenAI 0.20–0.60, hashing 0.0–0.50 (`SEMANTIC_RANGES`).
- **keyword**: the BM25 score relative to the best resume, 0..1.
- **skill_coverage**: required skills matched ÷ required skills. If the JD
  names no skills, its weight is shared between the other two.

All weights are constants at the top of `job_matcher.py`.

## Sample output

From `python job_matcher.py --jd-file job_descriptions/senior_python_ml.txt --include-filtered`
(offline hashing embedder; the JD text and lists are shortened here):

```json
{
  "job_description": "Senior Machine Learning Engineer\n\nAbout the role: ...",
  "top_matches": [
    {
      "candidate_name": "Priya Patel",
      "resume_path": "resumes/resume_priya_patel.docx",
      "match_score": 88,
      "matched_skills": ["Machine Learning", "NLP", "LLMs", "Python", "PyTorch", "TensorFlow",
                         "scikit-learn", "Docker", "Kubernetes", "RAG", "Hugging Face", "MLflow"],
      "relevant_excerpts": [
        "[Skills] Python, PyTorch, TensorFlow, Hugging Face Transformers, LLMs, NLP, scikit-learn, MLflow, Docker, Kubernetes, AWS SageMaker, SQL",
        "[Summary] Senior machine learning engineer with 9 years of Python experience shipping NLP and large language model systems to production.",
        "[Experience] Staff ML Engineer, Lumina AI (Jan 2021 – Present) - Fine-tuned transformer models in PyTorch ..."
      ],
      "reasoning": "Strong match: closest resume sections are Skills and Summary. Has 9/10 required skills (missing AWS), plus preferred RAG, Hugging Face and MLflow. Skills mentions NLP, LLMs, Python, PyTorch and TensorFlow; Summary mentions Machine Learning, NLP, LLMs and Python. Meets must-haves: 9.4 yrs Python (needs 5+); PhD (needs Bachelor's or higher). 9.4 years of total experience."
    }
  ],
  "filtered_out": [
    {
      "candidate_name": "Olivia Brown",
      "resume_path": "resumes/resume_olivia_brown.txt",
      "match_score": 35,
      "failed_requirements": ["1.4 yrs Python (needs 5+)"]
    }
  ]
}
```

## Tests

```bash
pytest -v
```

The suite covers section splitting, chunking, metadata extraction, the LLM
fallback, indexing and re-indexing, JD parsing, semantic/keyword/hybrid
search, must-have checks, scoring, output format and both CLIs. It needs no
network: OpenAI clients are faked, and ChromaDB runs on temporary directories
with the hashing embedder.

## Limitations

- **Alternatives:** "PyTorch or TensorFlow" in a JD makes both skills required.
- **Skill years** come from the roles that mention a skill (or an implied
  framework). A skill listed only in the Skills section is `unverified` for
  year requirements.
- **Keyword scores** are relative to the best resume in the index, so they
  rank candidates but don't measure absolute fit on their own.
- **The OpenAI semantic range** (0.20–0.60) is a heuristic. Adjust
  `SEMANTIC_RANGES` if scores look compressed or inflated for your data.
