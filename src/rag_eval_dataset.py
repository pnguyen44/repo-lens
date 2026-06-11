EVAL_CASES: list[dict[str, str | list[str]]] = [
    {
        "question": "What database does hyperfleet-api use?",
        "expected_sections": ["Architecture"],
        "expected_keywords": ["PostgreSQL", "GORM"],
    },
    {
        "question": "What language is hyperfleet-api written in?",
        "expected_sections": ["Architecture"],
        "expected_keywords": ["Go"],
    },
    {
        "question": "How do I run the integration tests?",
        "expected_sections": ["Development"],
        "expected_keywords": ["make test-integration"],
    },
    {
        "question": "What are the prerequisites to set up hyperfleet-api locally?",
        "expected_sections": ["Quick Start"],
        "expected_keywords": ["Go", "Podman", "PostgreSQL", "Make"],
    },
    {
        "question": "What port does the API run on?",
        "expected_sections": ["Quick Start"],
        "expected_keywords": ["8000"],
    },
    {
        "question": "What are the main cluster endpoints?",
        "expected_sections": ["API Resources"],
        "expected_keywords": ["clusters", "GET", "POST"],
    },
    {
        "question": "How do I search for clusters by label?",
        "expected_sections": ["Example Usage"],
        "expected_keywords": ["search", "labels"],
    },
    {
        "question": "What CLI subcommands does hyperfleet-api support?",
        "expected_sections": ["Development"],
        "expected_keywords": ["serve", "migrate", "version"],
    },
    {
        "question": "How is the project directory structured?",
        "expected_sections": ["Architecture"],
        "expected_keywords": ["cmd", "pkg", "handlers", "dao"],
    },
    {
        "question": "What is the first step after cloning the repo?",
        "expected_sections": ["Quick Start"],
        "expected_keywords": ["make generate-all"],
    },
]
