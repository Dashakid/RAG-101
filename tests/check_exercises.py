#!/usr/bin/env python3
"""
RAG-101 Workbook Validator
==========================
Checks environment setup, git health, notebook exercise completion,
pipeline smoke test, and evaluation results.

Usage:
    python tests/check_exercises.py env
    python tests/check_exercises.py git
    python tests/check_exercises.py notebook
    python tests/check_exercises.py pipeline      # uses API credits
    python tests/check_exercises.py eval
    python tests/check_exercises.py all
    python tests/check_exercises.py status
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# ── ANSI colour helpers ───────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}"); _failures.append(msg)
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):  print(f"  {CYAN}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n{'─' * len(msg)}")

_failures = []

ROOT = Path(__file__).parent.parent


# ══════════════════════════════════════════════════════════════
# CHECK: Environment
# ══════════════════════════════════════════════════════════════
def check_env():
    header("ENV — Packages & API Key")

    # Required packages
    required = [
        ("langchain",           "langchain"),
        ("langchain-community", "langchain_community"),
        ("langchain-openai",    "langchain_openai"),
        ("langchain-chroma",    "langchain_chroma"),
        ("chromadb",            "chromadb"),
        ("ragas",               "ragas"),
        ("datasets",            "datasets"),
        ("pandas",              "pandas"),
        ("matplotlib",          "matplotlib"),
        ("python-dotenv",       "dotenv"),
        ("beautifulsoup4",      "bs4"),
    ]

    all_imported = True
    for display, module in required:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "n/a")
            ok(f"{display:<25} {DIM}v{version}{RESET}")
        except ImportError:
            fail(f"{display} not installed — run: pip install {display}")
            all_imported = False

    # .env file
    print()
    env_path = ROOT / ".env"
    if env_path.exists():
        ok(".env file found")
        content = env_path.read_text()
        if "OPENAI_API_KEY" in content:
            # Extract value
            for line in content.splitlines():
                if line.startswith("OPENAI_API_KEY"):
                    val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    if val.startswith("sk-") and len(val) > 20 and "your" not in val.lower():
                        ok(f"OPENAI_API_KEY looks valid  ({val[:8]}...{val[-4:]})")
                    else:
                        fail("OPENAI_API_KEY found but looks like a placeholder — paste your real key")
        else:
            fail("OPENAI_API_KEY not found in .env — add: OPENAI_API_KEY=sk-...")
    else:
        fail(".env file missing — create it with: echo 'OPENAI_API_KEY=sk-...' > .env")

    return all_imported


# ══════════════════════════════════════════════════════════════
# CHECK: Git Health
# ══════════════════════════════════════════════════════════════
def check_git():
    header("GIT — Repository Health")

    def git(cmd):
        r = subprocess.run(f"git -C {ROOT} {cmd}", shell=True,
                           capture_output=True, text=True)
        return r.stdout.strip(), r.returncode

    # Git initialised
    if (ROOT / ".git").exists():
        ok("Git repository initialised")
    else:
        fail(".git not found — run: git init")
        return

    # .gitignore exists and contains .env
    gi = ROOT / ".gitignore"
    if gi.exists():
        ok(".gitignore exists")
        content = gi.read_text()
        if ".env" in content:
            ok(".env is in .gitignore  🔒")
        else:
            fail(".env is NOT in .gitignore — add it immediately to protect your API key")
    else:
        fail(".gitignore not found")

    # .env is not tracked
    tracked, _ = git("ls-files .env")
    if tracked:
        fail(".env is tracked by git — remove it: git rm --cached .env")
    else:
        ok(".env is not tracked by git  ✓")

    # Remote set
    remote, _ = git("remote get-url origin")
    if remote:
        ok(f"Remote origin → {remote}")
    else:
        fail("No remote set — run: git remote add origin https://github.com/Dashakid/RAG-101.git")

    # Commit count
    log, _ = git("log --oneline")
    commits = [l for l in log.splitlines() if l.strip()]
    count = len(commits)
    if count == 0:
        fail("No commits yet — make your first commit")
    elif count == 1:
        warn(f"Only 1 commit — keep going, checkpoints expected after each exercise")
        for c in commits:
            info(c)
    else:
        ok(f"{count} commit(s) found")
        for c in commits:
            info(c)

    # Checkpoint tags [RAG-N]
    checkpoint_commits = [c for c in commits if "[RAG-" in c]
    total_checkpoints  = 5
    print()
    info(f"Checkpoint commits: {len(checkpoint_commits)} / {total_checkpoints}")
    for c in checkpoint_commits:
        ok(f"  {c}")
    missing = total_checkpoints - len(checkpoint_commits)
    if missing > 0:
        warn(f"{missing} checkpoint(s) not yet committed")


# ══════════════════════════════════════════════════════════════
# CHECK: Notebook Exercise Completion
# ══════════════════════════════════════════════════════════════
def check_notebook():
    header("NOTEBOOK — Exercise Completion")

    nb_path = ROOT / "rag_eval_workflow.ipynb"
    if not nb_path.exists():
        fail("rag_eval_workflow.ipynb not found in project root")
        return

    ok("Notebook file found")
    nb = json.loads(nb_path.read_text())
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    info(f"Total code cells: {len(code_cells)}")

    # Patterns that mean a cell is still untouched
    PLACEHOLDER_PATTERNS = [
        "= None  # replace this line",
        "pass  # replace with your implementation",
        "# YOUR CODE HERE",
        "# YOUR SYSTEM PROMPT HERE",
    ]

    # Patterns that mean a cell IS implemented
    DONE_PATTERNS = {
        "Ex1 — Imports":       ["from dotenv import", "from langchain"],
        "Ex2 — Load doc":      ["WebBaseLoader(", "loader.load()"],
        "Ex3 — Chunking":      ["RecursiveCharacterTextSplitter(", "split_documents("],
        "Ex4 — Vector store":  ["OpenAIEmbeddings(", "Chroma.from_documents(", "as_retriever("],
        "Ex5 — RAG chain":     ["format_docs", "ChatOpenAI(", "rag_chain"],
        "Ex6 — Test queries":  ["my_questions", "rag_chain.invoke("],
        "Ex7 — Golden dataset":["questions = [", "ground_truths = [", "Dataset.from_dict("],
        "Ex8 — Evaluate":      ["evaluate(", "to_pandas("],
        "Ex9 — Visualize":     ["mean()", "to_csv("],
    }

    print()
    all_source = "\n".join(
        "".join(c["source"]) for c in code_cells
    )

    done_count = 0
    for exercise, patterns in DONE_PATTERNS.items():
        found = all(p in all_source for p in patterns)
        if found:
            ok(f"{exercise}")
            done_count += 1
        else:
            missing_pats = [p for p in patterns if p not in all_source]
            fail(f"{exercise}  {DIM}(missing: {missing_pats[0]}){RESET}")

    # Count remaining placeholder lines
    placeholder_count = sum(
        1 for c in code_cells
        for line in "".join(c["source"]).splitlines()
        if any(p in line for p in PLACEHOLDER_PATTERNS)
    )

    print()
    total = len(DONE_PATTERNS)
    pct = int(done_count / total * 100)
    bar_filled = int(done_count / total * 30)
    bar = f"[{'█' * bar_filled}{'░' * (30 - bar_filled)}]"
    print(f"  Progress: {bar} {done_count}/{total} exercises  ({pct}%)")

    if placeholder_count > 0:
        warn(f"{placeholder_count} placeholder line(s) still in notebook "
             f"— look for '= None  # replace this line'")
    else:
        ok("No placeholder lines remaining")


# ══════════════════════════════════════════════════════════════
# CHECK: Live Pipeline Smoke Test
# ══════════════════════════════════════════════════════════════
def check_pipeline():
    header("PIPELINE — Live Smoke Test  (uses API credits ~$0.01)")
    warn("This check makes real API calls — costs a small amount of credits")
    print()

    # Load env
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        fail("python-dotenv not installed")
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        fail("OPENAI_API_KEY not set — cannot run live test")
        return
    ok("API key loaded")

    # Imports
    try:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_chroma import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        ok("All imports successful")
    except ImportError as e:
        fail(f"Import error: {e}")
        return

    # Load document
    try:
        info("Loading Wikipedia article...")
        loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
        docs = loader.load()
        assert len(docs) > 0 and len(docs[0].page_content) > 500
        ok(f"Document loaded  ({len(docs[0].page_content):,} chars)")
    except Exception as e:
        fail(f"Document loading failed: {e}")
        return

    # Chunking
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 10
        ok(f"Chunking successful  ({len(chunks)} chunks)")
    except Exception as e:
        fail(f"Chunking failed: {e}")
        return

    # Embeddings + vector store (small subset to save cost)
    try:
        info("Embedding 10 chunks (cost: ~$0.001)...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vs = Chroma.from_documents(chunks[:10], embedding=embeddings,
                                   collection_name="rag_smoke_test")
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        results = retriever.invoke("What is RAG?")
        assert len(results) > 0
        ok(f"Vector store + retriever working  ({len(results)} results returned)")
    except Exception as e:
        fail(f"Embedding/vector store failed: {e}")
        return

    # RAG chain
    try:
        info("Running one chain invocation (cost: ~$0.005)...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only this context:\n{context}"),
            ("human", "{question}"),
        ])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        def fmt(docs): return "\n\n".join(d.page_content for d in docs)
        chain = (
            {"context": retriever | fmt, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        answer = chain.invoke("What problem does RAG solve?")
        assert isinstance(answer, str) and len(answer) > 20
        ok(f"RAG chain returned a valid answer  ({len(answer)} chars)")
        info(f"Sample: {answer[:120]}...")
    except Exception as e:
        fail(f"RAG chain failed: {e}")
        return

    print()
    ok("Full pipeline smoke test PASSED")


# ══════════════════════════════════════════════════════════════
# CHECK: Evaluation Results
# ══════════════════════════════════════════════════════════════
def check_eval():
    header("EVAL — RAGAS Results Validation")

    try:
        import pandas as pd
    except ImportError:
        fail("pandas not installed")
        return

    csv_path = ROOT / "rag_eval_results.csv"
    if not csv_path.exists():
        fail("rag_eval_results.csv not found")
        info("Run Exercise 9 in the notebook to generate it")
        return

    ok("rag_eval_results.csv found")
    df = pd.read_csv(csv_path)
    info(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")

    # Required columns
    required_cols = ["question", "answer", "faithfulness",
                     "answer_relevancy", "context_recall", "context_precision"]
    for col in required_cols:
        if col in df.columns:
            ok(f"Column present: {col}")
        else:
            fail(f"Missing column: {col}")

    # Score ranges
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    print()
    print(f"  {'Metric':<25} {'Mean':>6}  {'Min':>6}  {'Max':>6}  {'Status'}")
    print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")
    for col in metric_cols:
        if col not in df.columns:
            continue
        mean_val = df[col].mean()
        min_val  = df[col].min()
        max_val  = df[col].max()
        if not (0.0 <= mean_val <= 1.0):
            status = f"{RED}OUT OF RANGE{RESET}"
            fail(f"{col} mean {mean_val:.3f} is outside 0–1")
        elif mean_val >= 0.8:
            status = f"{GREEN}EXCELLENT{RESET}"
        elif mean_val >= 0.65:
            status = f"{YELLOW}GOOD{RESET}"
        else:
            status = f"{RED}NEEDS WORK{RESET}"
        print(f"  {col:<25} {mean_val:>6.3f}  {min_val:>6.3f}  {max_val:>6.3f}  {status}")

    print()
    if len(df) >= 10:
        ok(f"Dataset size adequate  ({len(df)} questions)")
    else:
        warn(f"Only {len(df)} questions — aim for 10+ for reliable metric averages")


# ══════════════════════════════════════════════════════════════
# STATUS DASHBOARD
# ══════════════════════════════════════════════════════════════
def status():
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  RAG-101 — Project Status Dashboard{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")

    checks = [
        ("Environment",  check_env),
        ("Git",          check_git),
        ("Notebook",     check_notebook),
        ("Eval Results", check_eval),
    ]

    summary = []
    for name, fn in checks:
        before = len(_failures)
        fn()
        after = len(_failures)
        new_fails = after - before
        summary.append((name, new_fails))

    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{'─'*55}")
    for name, fails in summary:
        symbol = f"{GREEN}✓ PASS{RESET}" if fails == 0 else f"{RED}✗ {fails} issue(s){RESET}"
        print(f"  {name:<20} {symbol}")

    total_fails = len(_failures)
    print(f"{'─'*55}")
    if total_fails == 0:
        print(f"  {GREEN}{BOLD}All checks passed!{RESET}")
    else:
        print(f"  {RED}{BOLD}{total_fails} total issue(s) to fix{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}\n")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
COMMANDS = {
    "env":      check_env,
    "git":      check_git,
    "notebook": check_notebook,
    "pipeline": check_pipeline,
    "eval":     check_eval,
    "all":      lambda: [check_env(), check_git(), check_notebook(), check_eval()],
    "status":   status,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[cmd]()

    if _failures and cmd != "status":
        print(f"\n{RED}✗ {len(_failures)} check(s) failed:{RESET}")
        for f in _failures:
            print(f"   • {f}")
        sys.exit(1)
    elif cmd != "status":
        print(f"\n{GREEN}✓ All checks passed!{RESET}")
