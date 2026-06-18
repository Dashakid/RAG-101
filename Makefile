# ══════════════════════════════════════════════════════════════
#  RAG-101 Makefile
#  Run `make help` to see all available targets.
# ══════════════════════════════════════════════════════════════

PYTHON  := /Users/dashakid/.pyenv/versions/lewagon/bin/python
SCRIPT  := tests/check_exercises.py
NOTEBOOK := rag_eval_workflow.ipynb

# Default target
.DEFAULT_GOAL := help

.PHONY: help check-env check-git check-notebook check-pipeline check-eval \
        test test-full status push clean

# ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  RAG-101 — Workbook Validator"
	@echo "  ════════════════════════════════════════════════"
	@echo ""
	@echo "  SAFE CHECKS  (no API calls, no cost)"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make check-env       Packages installed + API key present"
	@echo "  make check-git       Repo setup, .env protected, checkpoints"
	@echo "  make check-notebook  Exercise completion — TODOs remaining"
	@echo "  make check-eval      RAGAS results file valid + scores"
	@echo "  make test            Run all 4 safe checks above"
	@echo "  make status          Full dashboard summary of everything"
	@echo ""
	@echo "  LIVE CHECKS  (makes real API calls, costs ~\$$0.01)"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make check-pipeline  End-to-end smoke test with real LLM calls"
	@echo "  make test-full       All safe checks + live pipeline test"
	@echo ""
	@echo "  GIT HELPERS"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make push            Stage notebook + push to GitHub"
	@echo "  make log             Show commit history (oneline)"
	@echo ""
	@echo "  OTHER"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make clean           Remove generated output files"
	@echo ""

# ── Individual checks ─────────────────────────────────────────
check-env:
	@echo "\n▶  Checking environment...\n"
	@$(PYTHON) $(SCRIPT) env

check-git:
	@echo "\n▶  Checking git health...\n"
	@$(PYTHON) $(SCRIPT) git

check-notebook:
	@echo "\n▶  Checking notebook exercise completion...\n"
	@$(PYTHON) $(SCRIPT) notebook

check-pipeline:
	@echo "\n▶  Running live pipeline smoke test...\n"
	@$(PYTHON) $(SCRIPT) pipeline

check-eval:
	@echo "\n▶  Validating evaluation results...\n"
	@$(PYTHON) $(SCRIPT) eval

# ── Combined targets ──────────────────────────────────────────
test: check-env check-git check-notebook check-eval
	@echo "\n✓  All safe checks complete.\n"

test-full: check-env check-git check-notebook check-pipeline check-eval
	@echo "\n✓  Full test suite complete (including live API calls).\n"

status:
	@$(PYTHON) $(SCRIPT) status

# ── Git helpers ───────────────────────────────────────────────
push:
	@echo "\n▶  Checking .env is not staged...\n"
	@if git ls-files --error-unmatch .env 2>/dev/null; then \
		echo "🚨  ERROR: .env is tracked by git. Run: git rm --cached .env"; \
		exit 1; \
	fi
	@git add $(NOTEBOOK) README.md .gitignore Makefile tests/
	@git status --short
	@echo ""
	@read -p "Commit message: " MSG; \
	git commit -m "$$MSG"; \
	git push origin main
	@echo "\n✓  Pushed to GitHub.\n"

log:
	@git log --oneline --color=always

# ── Clean ─────────────────────────────────────────────────────
clean:
	@echo "Removing generated output files..."
	@rm -f rag_eval_results.csv rag_eval_results.json
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓  Clean."
