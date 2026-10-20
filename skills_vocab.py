"""Skills vocabulary shared by the resume indexer and the job matcher.

Maps canonical skill names to the spellings that appear in resumes and job
descriptions, so that "sklearn", "scikit learn" and "scikit-learn" all count as
the same skill on both sides of a match.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Canonical name -> aliases (lower-case; the canonical name itself is always an alias).
SKILLS: dict[str, list[str]] = {
    # Languages
    "Python": ["python3"],
    "Java": [],
    "JavaScript": ["js", "ecmascript"],
    "TypeScript": [],
    "Go": ["golang"],
    "Rust": [],
    "C++": ["cpp"],
    "C#": ["csharp", "c sharp"],
    "C": [],
    "Scala": [],
    "Kotlin": [],
    "Swift": [],
    "Objective-C": ["objective c", "objc"],
    "Ruby": [],
    "PHP": [],
    "R": [],
    "SQL": [],
    "Bash": ["shell scripting", "shell"],
    "Dart": [],
    "MATLAB": [],
    "HTML": ["html5"],
    "CSS": ["css3"],
    # Web and backend frameworks
    "Django": [],
    "Django REST Framework": ["drf", "django rest"],
    "Flask": [],
    "FastAPI": ["fast api"],
    "Spring": [],
    "Spring Boot": ["springboot"],
    "Hibernate": [],
    "Node.js": ["nodejs", "node js", "node"],
    "Express": ["express.js", "expressjs"],
    "GraphQL": [],
    "REST APIs": ["rest", "rest api", "restful", "restful apis", "rest apis"],
    "Microservices": ["microservice", "micro-services"],
    "Celery": [],
    # Frontend and mobile
    "React": ["react.js", "reactjs"],
    "Next.js": ["nextjs"],
    "Redux": [],
    "Angular": ["angularjs"],
    "Vue.js": ["vue", "vuejs"],
    "Tailwind CSS": ["tailwind"],
    "SwiftUI": [],
    "Jetpack Compose": [],
    "Flutter": [],
    "Android": [],
    "iOS": [],
    "Firebase": [],
    "Figma": [],
    # Data and ML
    "Machine Learning": ["ml", "machine-learning"],
    "Deep Learning": ["deep-learning"],
    "NLP": ["natural language processing"],
    "Computer Vision": [],
    "LLMs": ["llm", "large language model", "large language models"],
    "Generative AI": ["genai", "gen ai"],
    "RAG": ["retrieval-augmented generation", "retrieval augmented generation"],
    "MLOps": [],
    "Statistics": ["statistical modeling", "statistical modelling"],
    "A/B Testing": ["a/b tests", "ab testing"],
    "pandas": [],
    "NumPy": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "PyTorch": ["torch"],
    "TensorFlow": ["tensorflow serving"],
    "Keras": [],
    "Hugging Face": ["huggingface", "hugging face transformers", "transformers"],
    "XGBoost": [],
    "OpenCV": [],
    "CUDA": [],
    "MLflow": [],
    "Kubeflow": [],
    "AWS SageMaker": ["sagemaker", "amazon sagemaker"],
    "Spark": ["apache spark", "pyspark"],
    "Hadoop": [],
    "Airflow": ["apache airflow"],
    "Kafka": ["apache kafka"],
    "Databricks": [],
    "Snowflake": [],
    "dbt": [],
    "AWS Glue": [],
    "ETL": ["elt"],
    "Tableau": [],
    "Power BI": ["powerbi"],
    # Databases
    "PostgreSQL": ["postgres"],
    "MySQL": [],
    "SQLite": [],
    "MongoDB": ["mongo"],
    "Redis": [],
    "Oracle": [],
    "Elasticsearch": ["elastic search"],
    "Cassandra": [],
    "DynamoDB": [],
    # Cloud, DevOps and tooling
    "AWS": ["amazon web services"],
    "GCP": ["google cloud", "google cloud platform"],
    "Azure": ["microsoft azure"],
    "Docker": [],
    "Kubernetes": ["k8s"],
    "Helm": [],
    "Terraform": [],
    "Ansible": [],
    "Jenkins": [],
    "GitHub Actions": [],
    "GitLab": ["gitlab ci"],
    "CI/CD": ["ci cd", "continuous integration", "continuous delivery"],
    "Git": [],
    "Linux": [],
    "Prometheus": [],
    "Grafana": [],
    "Maven": [],
    # Testing
    "pytest": [],
    "JUnit": [],
    "Jest": [],
    "Cypress": [],
    "Selenium": [],
    "XCTest": [],
    # Practices
    "Agile": [],
    "Scrum": [],
}

# Aliases that are ordinary words: in free text they only count when spelled exactly
# like this, so "react to incidents" is not read as React but "Staff ML Engineer" is ML.
CASE_SENSITIVE = {
    "Go", "Rust", "Swift", "Ruby", "Dart", "Spring", "Flask", "Express", "React", "Redux",
    "Angular", "Spark", "Helm", "Jest", "Oracle", "Agile", "Scrum", "Snowflake", "Android",
    "Keras", "ML", "REST", "JS",
}
_CASE_SENSITIVE_BY_ALIAS = {spelling.lower(): spelling for spelling in CASE_SENSITIVE}
# Too short or ambiguous to find in prose at all; only recognised as an item of a skills list.
LIST_ONLY = {"c", "r", "shell", "node", "torch", "transformers", "elt"}

_ALIAS_TO_SKILL: dict[str, str] = {}
for _canonical, _aliases in SKILLS.items():
    for _alias in [_canonical.lower(), *_aliases]:
        _ALIAS_TO_SKILL[_alias] = _canonical

_VERSION_SUFFIX_RE = re.compile(r"\s+v?\d+(?:\.\d+)*\+?$")


def normalize_skill(raw: str) -> str | None:
    """Return the canonical name for ``raw`` (e.g. ``"sklearn"`` -> ``"scikit-learn"``).

    Trailing versions are ignored (``"Java 17"`` -> ``"Java"``). Returns
    ``None`` when the skill is not in the vocabulary.
    """
    key = " ".join(raw.strip().lower().split())
    if key in _ALIAS_TO_SKILL:
        return _ALIAS_TO_SKILL[key]
    key = _VERSION_SUFFIX_RE.sub("", key)
    return _ALIAS_TO_SKILL.get(key)


@lru_cache(maxsize=1)
def _free_text_patterns() -> tuple[re.Pattern, re.Pattern]:
    insensitive, sensitive = [], []
    for alias, canonical in _ALIAS_TO_SKILL.items():
        if alias in LIST_ONLY:
            continue
        if alias in _CASE_SENSITIVE_BY_ALIAS:
            sensitive.append(_CASE_SENSITIVE_BY_ALIAS[alias])
        else:
            insensitive.append(alias)

    def build(aliases: list[str], flags: int) -> re.Pattern:
        # Longest first so "Spring Boot" wins over "Spring" at the same position.
        alternatives = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
        return re.compile(rf"(?<![\w+#.])(?:{alternatives})(?![\w+#]|\.\w)", flags)

    return build(insensitive, re.IGNORECASE), build(sensitive, 0)


def find_skills(text: str) -> list[str]:
    """Find vocabulary skills mentioned anywhere in free text, in order of first mention."""
    found: list[tuple[int, int, str]] = []
    for pattern in _free_text_patterns():
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), _ALIAS_TO_SKILL[match.group(0).lower()]))
    # Drop matches nested inside a longer match from the other pattern.
    found.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    skills, covered_until = [], -1
    for start, end, skill in found:
        if start < covered_until:
            continue
        covered_until = end
        if skill not in skills:
            skills.append(skill)
    return skills


# Using a framework implies using its language: years with Django count as years with Python.
IMPLIED_SKILLS = {
    "Django": "Python", "Django REST Framework": "Python", "Flask": "Python",
    "FastAPI": "Python", "Celery": "Python", "pytest": "Python", "pandas": "Python",
    "NumPy": "Python", "scikit-learn": "Python", "PyTorch": "Python", "Airflow": "Python",
    "Spring": "Java", "Spring Boot": "Java", "Hibernate": "Java", "JUnit": "Java",
    "Maven": "Java", "React": "JavaScript", "Next.js": "JavaScript", "Redux": "JavaScript",
    "Node.js": "JavaScript", "Express": "JavaScript", "Angular": "TypeScript",
    "Vue.js": "JavaScript", "Jest": "JavaScript", "SwiftUI": "Swift", "XCTest": "Swift",
    "Jetpack Compose": "Kotlin", "Flutter": "Dart",
}


def with_implied(skills: list[str]) -> list[str]:
    """``skills`` followed by the languages they imply that are not already listed."""
    result = list(skills)
    for skill in skills:
        implied = IMPLIED_SKILLS.get(skill)
        if implied and implied not in result:
            result.append(implied)
    return result
