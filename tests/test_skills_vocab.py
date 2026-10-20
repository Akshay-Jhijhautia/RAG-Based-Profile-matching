import pytest

import skills_vocab


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sklearn", "scikit-learn"),
        ("Scikit Learn", "scikit-learn"),
        ("K8s", "Kubernetes"),
        ("Java 17", "Java"),
        ("python3", "Python"),
        ("  apache   spark ", "Spark"),
        ("C++", "C++"),
        ("R", "R"),
        ("WCAG accessibility", None),
    ],
)
def test_normalize_skill(raw, expected):
    assert skills_vocab.normalize_skill(raw) == expected


def test_find_skills_prefers_longest_match():
    assert skills_vocab.find_skills("Built Spring Boot services") == ["Spring Boot"]
    assert skills_vocab.find_skills("Django REST Framework APIs") == ["Django REST Framework"]


def test_find_skills_respects_word_boundaries():
    assert skills_vocab.find_skills("JavaScript and TypeScript") == ["JavaScript", "TypeScript"]
    assert skills_vocab.find_skills("Worked at Google on Gopher tools") == []


def test_find_skills_case_sensitive_words():
    assert skills_vocab.find_skills("Quick to react to incidents with swift fixes") == []
    assert skills_vocab.find_skills("React and Swift apps written in Go") == ["React", "Swift", "Go"]
    assert skills_vocab.find_skills("Staff ML Engineer") == ["Machine Learning"]


def test_find_skills_list_only_aliases_ignored_in_prose():
    assert skills_vocab.find_skills("Grade C student, wrote C++ and C# code") == ["C++", "C#"]


def test_find_skills_symbols_and_punctuation():
    assert skills_vocab.find_skills("Uses Node.js, CI/CD and Python.") == ["Node.js", "CI/CD", "Python"]
    assert skills_vocab.find_skills("retrieval-augmented generation with LLMs") == ["RAG", "LLMs"]


def test_with_implied_adds_languages_once():
    assert skills_vocab.with_implied(["Django", "Flask", "React"]) == [
        "Django", "Flask", "React", "Python", "JavaScript"
    ]
    assert skills_vocab.with_implied(["Python", "Django"]) == ["Python", "Django"]
