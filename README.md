# RAG 101

End-to-end Retrieval-Augmented Generation pipeline built from scratch as a portfolio project.

## Stack
| Layer | Tool |
|-------|------|
| Framework | LangChain (LCEL) |
| Vector DB | ChromaDB (local) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Evaluation | RAGAS |

## Project Structure
```
RAG-101/
├── rag_eval_workflow.ipynb   # Main guided workbook
├── tests/
│   └── check_exercises.py    # Validation suite
├── Makefile                  # Run checks with `make`
├── .env                      # API keys (not committed)
├── .gitignore
└── README.md
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Dashakid/RAG-101.git
cd RAG-101

# 2. Create your .env
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Validate your environment
make check-env

# 4. Open the workbook and follow exercises 1–9
# 5. After each exercise, run the corresponding checkpoint:
make check-notebook

# 6. Run a full pipeline smoke test (uses API credits ~$0.01)
make check-pipeline
```

## Validation Commands

| Command | What it checks |
|---------|---------------|
| `make help` | Show all available targets |
| `make check-env` | Packages installed, API key present |
| `make check-git` | Repo setup, .env protected, checkpoints committed |
| `make check-notebook` | Exercise completion — how many TODOs remain |
| `make check-pipeline` | Live end-to-end smoke test (costs API credits) |
| `make check-eval` | RAGAS results file valid with expected columns |
| `make test` | All checks except live API calls |
| `make test-full` | Everything including live pipeline test |
| `make status` | Summary dashboard of all checks |

## Commit History Pattern

Each checkpoint commit follows this format:
```
[RAG-N] Short description

Exercises: completed list
Status: ✓ all checks pass
Notes: what you learned
```

## Author
Dashakid — [github.com/Dashakid](https://github.com/Dashakid)
