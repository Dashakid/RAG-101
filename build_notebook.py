"""Generates rag_eval_workflow.ipynb from cell definitions."""
import json, textwrap

def md(source): return {"cell_type":"markdown","metadata":{},"source":source.lstrip("\n")}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.lstrip("\n")}

cells = []

# ─── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md("""
# 📖 RAG 101 — Guided Workbook

**Author:** `[Your Name]`  **Date:** `[Date]`

> **How to use this workbook**
> - Read every markdown cell before writing any code — they explain the *why*
> - Fill in every `# YOUR CODE HERE` block
> - Run the `# ── SELF CHECK ──` block at the bottom of each cell to verify your work
> - Use `make check-notebook` in the terminal any time to see your overall progress
> - Commit at every 📌 **Git Checkpoint** — those cells write the commit for you

---

## What You Are Building

```
[Document / Web Page]
        │
        ▼  Exercise 2 — Load
  list[Document]
        │
        ▼  Exercise 3 — Chunk
  list[Chunk]  (512 chars, 64 overlap)
        │
        ▼  Exercise 4 — Embed + Index
  ChromaDB Vector Store
        │
  User Query ──► Retriever (top-4 chunks)
                        │
        ▼  Exercise 5 — RAG Chain
  Prompt Template  +  gpt-4o-mini
                        │
                    Answer
                        │
        ▼  Exercises 7–9 — Evaluate
  RAGAS Scores  ──►  CSV Export
```

**Stack:** LangChain · ChromaDB · OpenAI · RAGAS · HuggingFace Datasets

**Prerequisite:** An OpenAI API key — get one at https://platform.openai.com
"""))

# ─── GIT SETUP ────────────────────────────────────────────────────────────────
cells.append(md("""
---

## 🗂️ Git Setup

Run the cell below **once** before you start. It wires up your local folder to the
GitHub repo `Dashakid/RAG-101` and creates a `.gitignore` that protects your API key.

> 🔒 **Rule:** Never let `.env` appear in `git status` as a staged or tracked file.
> The setup cell and every checkpoint cell check this automatically.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# GIT SETUP — Run once, then never again
# ════════════════════════════════════════════════════════════════
import subprocess, os
from pathlib import Path

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout.strip())
    if r.stderr.strip(): print(r.stderr.strip())

# Confirm .env is in .gitignore
gi = Path(".gitignore").read_text() if Path(".gitignore").exists() else ""
assert ".env" in gi, "ERROR: .env is not in .gitignore — check the file before proceeding"
print("✓ .env is protected by .gitignore")

# Show remote
remote = subprocess.run("git remote get-url origin", shell=True,
                        capture_output=True, text=True).stdout.strip()
print(f"✓ Remote: {remote}")

# Show current branch
branch = subprocess.run("git branch --show-current", shell=True,
                        capture_output=True, text=True).stdout.strip()
print(f"✓ Branch: {branch}")
print("\\nGit setup confirmed. Ready to start exercises.")
"""))

# ─── EXERCISE 0 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 0 — Install Dependencies

**Instructions:**
1. Just run this cell — nothing to write yet
2. Read the package list and look up any package you don't recognise
3. The `-U` flag upgrades to latest versions; `-q` keeps output clean

**Expected output:** `✓ Packages ready.`
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 0 — Install & Upgrade All Dependencies
# ════════════════════════════════════════════════════════════════

%pip install -U -q pip

%pip install -U -q \\
    langchain \\
    langchain-community \\
    langchain-openai \\
    langchain-chroma \\
    chromadb \\
    ragas \\
    datasets \\
    pandas \\
    matplotlib \\
    seaborn \\
    python-dotenv \\
    beautifulsoup4

print("✓ Packages ready. Move on to Exercise 1.")
"""))

# ─── EXERCISE 1 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 1 — Imports & API Key

### Concept: why we import at the top
Python needs to know which external libraries you're using before you call them.
Grouping all imports at the top of the notebook is a best practice — it makes
dependencies visible and prevents "name not defined" errors halfway through.

### What to import

| What you need | Package path | Class / function |
|--------------|-------------|-----------------|
| Load web pages | `langchain_community.document_loaders` | `WebBaseLoader` |
| Split text | `langchain.text_splitter` | `RecursiveCharacterTextSplitter` |
| Create embeddings | `langchain_openai` | `OpenAIEmbeddings` |
| Chat LLM | `langchain_openai` | `ChatOpenAI` |
| Local vector DB | `langchain_chroma` | `Chroma` |
| Prompt template | `langchain_core.prompts` | `ChatPromptTemplate` |
| Parse LLM output | `langchain_core.output_parsers` | `StrOutputParser` |
| Pass data through | `langchain_core.runnables` | `RunnablePassthrough` |
| RAGAS runner | `ragas` | `evaluate` |
| RAGAS metrics | `ragas.metrics` | `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision` |
| Dataset format | `datasets` | `Dataset` |
| Data analysis | `pandas` | import as `pd` |

### API Key

Create a `.env` file in this folder containing one line:
```
OPENAI_API_KEY=sk-your-key-here
```
Then call `load_dotenv()` to read it. Import `load_dotenv` from `dotenv`.

**Instructions:**
1. Write one `from ... import ...` line for each row in the table above
2. Call `load_dotenv()` to load your `.env` file
3. Read `OPENAI_API_KEY` from the environment with `os.getenv(...)`
4. Run the self-check — it will tell you exactly which imports are missing

**Expected output:** `✓ All imports found!` and `✓ API key loaded`
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 1 — Imports & API Key Configuration
# ════════════════════════════════════════════════════════════════

import os
import warnings
warnings.filterwarnings("ignore")

# ── YOUR CODE HERE ────────────────────────────────────────────
# 1. Import load_dotenv
#    from dotenv import ...

# 2. Import the LangChain classes (one line per class)
#    from langchain_community.document_loaders import ...
#    from langchain.text_splitter import ...
#    from langchain_openai import ...  (two classes on one line is fine)
#    from langchain_chroma import ...
#    from langchain_core.prompts import ...
#    from langchain_core.output_parsers import ...
#    from langchain_core.runnables import ...

# 3. Import RAGAS evaluate + 4 metrics

# 4. Import Dataset and pandas


# ── SELF CHECK ───────────────────────────────────────────────
try:
    _ = [WebBaseLoader, RecursiveCharacterTextSplitter,
         OpenAIEmbeddings, ChatOpenAI, Chroma,
         ChatPromptTemplate, StrOutputParser, RunnablePassthrough,
         evaluate, faithfulness, answer_relevancy,
         context_recall, context_precision,
         Dataset, pd]
    print("✓ All imports found!")
except NameError as e:
    print(f"✗ Missing: {e}  — add the import above and re-run")

# ── YOUR CODE HERE ────────────────────────────────────────────
# 5. Call load_dotenv()

# 6. Store your API key: api_key = os.getenv("OPENAI_API_KEY")


# ── SELF CHECK ───────────────────────────────────────────────
try:
    if api_key and api_key.startswith("sk-") and "your" not in api_key.lower():
        print(f"✓ API key loaded  ({api_key[:8]}...{api_key[-4:]})")
    else:
        print("✗ API key missing or looks like a placeholder — check your .env file")
except NameError:
    print("✗ api_key not defined — did you write step 6 above?")
"""))

# ─── EXERCISE 2 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 2 — Load a Document

### Concept: Document Loaders

LangChain wraps many data sources behind a single interface. Every loader exposes
one method: `.load()` which returns a `list[Document]`.

A `Document` object has two attributes:
```python
doc.page_content   # the raw text string
doc.metadata       # dict — source URL, page number, filename, etc.
```

We are loading the Wikipedia article on *Retrieval-Augmented Generation* — meta
and perfect for demonstrating what RAG can do.

**URL:** `https://en.wikipedia.org/wiki/Retrieval-augmented_generation`

### Loader options (for your future projects)

| Your data source | Loader class |
|-----------------|-------------|
| Single PDF | `PyPDFLoader("file.pdf")` |
| Folder of PDFs | `DirectoryLoader("./data", glob="**/*.pdf")` |
| Plain text file | `TextLoader("file.txt")` |
| Website / blog | `WebBaseLoader("https://...")` |
| Notion database | `NotionDBLoader(...)` |

**Instructions:**
1. Create a `WebBaseLoader` instance — pass `SOURCE_URL` as the argument
2. Call `.load()` and store the result in `documents`
3. Print: number of documents, total character count, first 500 chars of `documents[0].page_content`

**Expected output:** 1 document, several thousand characters of Wikipedia text.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 2 — Load the Source Document
# ════════════════════════════════════════════════════════════════

SOURCE_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

# ── YOUR CODE HERE ────────────────────────────────────────────
# Step 1: Create a WebBaseLoader — pass SOURCE_URL as the argument
#   loader = WebBaseLoader(...)
loader = None  # replace this line

# Step 2: Call loader.load() — store result in `documents`
#   documents = loader.load()
documents = None  # replace this line

# Step 3: Print stats
#   a) len(documents)
#   b) total chars: sum(len(d.page_content) for d in documents)
#   c) documents[0].page_content[:500]


# ── SELF CHECK ───────────────────────────────────────────────
assert documents not in (None, []), "✗ documents is empty — did you call loader.load()?"
assert len(documents) > 0,          "✗ No documents returned — check the URL"
assert len(documents[0].page_content) > 500, \\
    "✗ Document is too short — something went wrong with loading"
print(f"✓ Document loaded  ({len(documents[0].page_content):,} characters)")
"""))

# ─── EXERCISE 3 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 3 — Chunk the Document

### Concept: Why We Chunk

An LLM has a fixed context window (e.g. 128k tokens for GPT-4o). More importantly,
embedding a 50,000-character article as a single vector produces a vague "average"
of everything in it — too blurry for precise retrieval.

Chunking breaks the document into short, focused passages so each embedding
represents one specific idea. Overlapping chunks ensures nothing is cut off mid-thought.

```
Chunk 1: [═══════════════════════════════]
Chunk 2:                    [═══════════════════════════════]
                  ◄── overlap 64 chars ──►
```

### RecursiveCharacterTextSplitter

Splits on natural boundaries in this priority order:
`\\n\\n` → `\\n` → `. ` → ` ` → individual characters

This preserves semantic coherence far better than slicing at a fixed character count.

**Parameters to use:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `chunk_size` | `512` | ~2–3 sentences — specific enough for precise retrieval |
| `chunk_overlap` | `64` | ~12% — prevents boundary cut-offs |
| `length_function` | `len` | measure in characters (not tokens) |

**Instructions:**
1. Create a `RecursiveCharacterTextSplitter` with the values in the table above
2. Call `.split_documents(documents)` — store result in `chunks`
3. Print: total chunks, average chunk length, content of the first chunk

**Expected output:** 50–150 chunks depending on the article length.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 3 — Chunk the Document
# ════════════════════════════════════════════════════════════════

# ── YOUR CODE HERE ────────────────────────────────────────────
# Step 1: Create the text splitter
#   text_splitter = RecursiveCharacterTextSplitter(
#       chunk_size=...,
#       chunk_overlap=...,
#       length_function=...,
#   )
text_splitter = None  # replace this line

# Step 2: Split the documents
#   chunks = text_splitter.split_documents(documents)
chunks = None  # replace this line

# Step 3: Print stats
#   - Total chunks:        len(chunks)
#   - Average chunk length: sum(len(c.page_content) for c in chunks) / len(chunks)
#   - First chunk preview: chunks[0].page_content[:300]


# ── SELF CHECK ───────────────────────────────────────────────
assert chunks not in (None, []),      "✗ chunks is empty — did you call split_documents()?"
assert len(chunks) > 10,              f"✗ Only {len(chunks)} chunks — check chunk_size parameter"
assert all(len(c.page_content) <= 600 for c in chunks), \\
    "✗ Some chunks exceed 600 chars — double-check chunk_size=512"
avg = sum(len(c.page_content) for c in chunks) / len(chunks)
print(f"✓ {len(chunks)} chunks created  (avg {avg:.0f} chars each)")
"""))

# ─── GIT CHECKPOINT 1 ────────────────────────────────────────────────────────
cells.append(md("""
---

## 📌 Git Checkpoint 1 — Document Loaded & Chunked

**Commit before moving on.**  
Chunking strategy is a key design decision — committing it here lets you
experiment with different `chunk_size` values later and diff the results.

**Fill in the `NOTES` string below, then run the cell.**
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# GIT CHECKPOINT 1
# ════════════════════════════════════════════════════════════════
# Edit NOTES with YOUR observations, then run.

NOTES = \"\"\"
Exercises: 1 (imports), 2 (document loading), 3 (chunking)
Status: ✓ / ⚠ / ✗   ← delete two

Chunk count     : ???   ← fill in len(chunks)
Avg chunk length: ???   ← fill in average chars
Chunk size used : 512   overlap: 64

Notes (what you learned / what was confusing):
-
-
\"\"\"

import subprocess

# Safety: block commit if .env is staged
staged = subprocess.run("git diff --cached --name-only",
                        shell=True, capture_output=True, text=True).stdout
assert ".env" not in staged, "🚨  .env is staged! Run: git reset HEAD .env  then retry"
print("✓ .env safety check passed")

subprocess.run("git add rag_eval_workflow.ipynb", shell=True)
subprocess.run(["git", "commit", "-m",
    f"[RAG-1] Document loaded and chunked\\n\\n{NOTES.strip()}"])
print("\\n✓ Committed! Run `make check-git` to verify.")
print("  Push when ready:  git push origin main")
"""))

# ─── EXERCISE 4 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 4 — Embeddings & Vector Store

### Concept: What Is an Embedding?

An embedding model converts any string of text into a fixed-length list of numbers
(a vector). Texts that mean similar things produce vectors that are close together
in that numeric space.

```
"What is RAG?"    →  [0.021, -0.134, 0.872, ..., 0.003]   ← 1536 numbers
"RAG overview"    →  [0.019, -0.130, 0.869, ..., 0.001]   ← very close!
"Chocolate cake"  →  [-0.412, 0.823, -0.011, ..., 0.234]  ← far away
```

Retrieval works by embedding the user's question and finding the chunk vectors
nearest to it — those are the most relevant passages.

### Embedding model

```python
OpenAIEmbeddings(model="text-embedding-3-small")
```
1536 dimensions, fast, cheap (~$0.02 per 1M tokens).

### ChromaDB

Runs **locally** — no cloud account, no network latency. Stores your vectors and
runs similarity search in milliseconds.

```python
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_wiki",
)
```

After building the store, wrap it in a **retriever**:
```python
vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```
`k=4` means return the 4 most relevant chunks per query.

**Instructions:**
1. Initialize `OpenAIEmbeddings` with `model="text-embedding-3-small"`
2. Call `Chroma.from_documents(...)` with your chunks and embeddings — store as `vector_store`
3. Create a retriever with `search_type="similarity"` and `k=4`
4. Test it: call `retriever.invoke("What problem does RAG solve?")` and print each chunk

> ⏱ Step 2 makes API calls to embed all chunks — takes ~10–30 seconds.

**Expected output:** 4 chunks returned, each showing relevant text about RAG.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 4 — Embeddings & Vector Store
# ════════════════════════════════════════════════════════════════

# ── YOUR CODE HERE ────────────────────────────────────────────
# Step 1: Initialize the embedding model
#   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = None  # replace this line

# Step 2: Build the Chroma vector store from your chunks
#   This embeds every chunk and stores the vectors — run once per dataset.
#   vector_store = Chroma.from_documents(
#       documents=chunks,
#       embedding=embeddings,
#       collection_name="rag_wiki",
#   )
print("Embedding chunks — this takes ~10-30 seconds...")
vector_store = None  # replace this line

# Step 3: Create a retriever
#   retriever = vector_store.as_retriever(
#       search_type="similarity",
#       search_kwargs={"k": 4},
#   )
retriever = None  # replace this line

# Step 4: Test the retriever
TEST_QUERY = "What problem does RAG solve?"
# results = retriever.invoke(TEST_QUERY)
# for i, doc in enumerate(results, 1):
#     print(f"  Chunk {i}: {doc.page_content[:150].strip()}...")


# ── SELF CHECK ───────────────────────────────────────────────
assert embeddings  is not None, "✗ embeddings not set"
assert vector_store is not None, "✗ vector_store not built — did you call Chroma.from_documents()?"
assert retriever   is not None, "✗ retriever not created — did you call .as_retriever()?"

test_results = retriever.invoke("What is retrieval augmented generation?")
assert len(test_results) == 4, \\
    f"✗ Expected 4 results, got {len(test_results)} — check search_kwargs={{\"k\": 4}}"
assert all(hasattr(r, "page_content") for r in test_results), \\
    "✗ Results don't look like Document objects — check Chroma.from_documents arguments"
print(f"\\n✓ Vector store ready — {vector_store._collection.count()} vectors indexed")
print(f"✓ Retriever returns {len(test_results)} chunks per query")
"""))

# ─── EXERCISE 5 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 5 — Build the RAG Chain

### Concept: LangChain Expression Language (LCEL)

LCEL lets you chain components together with the `|` pipe operator.
Each component receives the output of the previous one.

The full data flow:

```
User Question (string)
   │
   ├─────────────────────────────────────────────┐
   ▼                                             ▼
retriever.invoke(question)           RunnablePassthrough()
   │                                             │
   ▼                                             │
[Doc1, Doc2, Doc3, Doc4]                         │
   │                                             │
   ▼                                             │
format_docs(docs) → single string                │
   │                                             │
   └──────────── ChatPromptTemplate ◄────────────┘
                        │
                  ChatOpenAI LLM
                        │
                  StrOutputParser
                        │
                   Final Answer (string)
```

### Your system prompt must:
- Tell the LLM to answer **only from the provided context**
- Include a `{context}` placeholder (filled by `format_docs`)
- Give a clear fallback: *"If the answer is not in the context, say 'I don't know.'"*

### Assembling the chain with LCEL

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**Instructions:**
1. Write `format_docs(docs)` — joins each doc's `.page_content` with `"\\n\\n"` as separator
2. Write your `SYSTEM_TEMPLATE` string — must contain `{context}`
3. Build `prompt` with `ChatPromptTemplate.from_messages([("system", ...), ("human", "{question}")])`
4. Initialize `llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)`
5. Assemble `rag_chain` using the LCEL pipe syntax above
6. Test it: run `rag_chain.invoke("What is retrieval-augmented generation?")`

**Expected output:** A clear, grounded answer about RAG based on the Wikipedia article.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 5 — Assemble the RAG Chain
# ════════════════════════════════════════════════════════════════

# ── YOUR CODE HERE ────────────────────────────────────────────
# Step 1: Write the format_docs helper function
#   Input:  list of Document objects
#   Output: single string — each doc's .page_content joined by "\\n\\n"
def format_docs(docs):
    pass  # replace with your implementation


# Step 2: Write the system prompt template
#   Must include {context} — the retrieved chunks go here
SYSTEM_TEMPLATE = \"\"\"
# YOUR SYSTEM PROMPT HERE
# Must include a {context} placeholder
# Must tell the LLM to answer ONLY from the context
# Must include a fallback for when the answer isn't in the context
\"\"\"


# Step 3: Build the prompt template
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", SYSTEM_TEMPLATE),
#       ("human", "{question}"),
#   ])
prompt = None  # replace this line


# Step 4: Initialize the LLM
#   temperature=0 → deterministic answers (important for reproducible eval)
#   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = None  # replace this line


# Step 5: Assemble the chain
#   rag_chain = (
#       {"context": retriever | format_docs, "question": RunnablePassthrough()}
#       | prompt
#       | llm
#       | StrOutputParser()
#   )
rag_chain = None  # replace this line


# Step 6: Test it
# test_answer = rag_chain.invoke("What is retrieval-augmented generation?")
# print(test_answer)


# ── SELF CHECK ───────────────────────────────────────────────
# Test format_docs
sample_docs = retriever.invoke("what is RAG?")
formatted = format_docs(sample_docs)
assert isinstance(formatted, str) and len(formatted) > 100, \\
    "✗ format_docs should return a non-empty string joined from doc.page_content"
print("✓ format_docs works")

assert "{context}" in SYSTEM_TEMPLATE, \\
    "✗ SYSTEM_TEMPLATE must contain {context} — the retrieved chunks go here"
assert prompt    is not None, "✗ prompt not built — call ChatPromptTemplate.from_messages(...)"
assert llm       is not None, "✗ llm not initialized — call ChatOpenAI(...)"
assert rag_chain is not None, "✗ rag_chain not assembled — use the LCEL pipe syntax"

print("Running live chain test...")
answer = rag_chain.invoke("What is retrieval-augmented generation?")
assert isinstance(answer, str) and len(answer) > 20, \\
    "✗ Chain returned empty or unexpected output"
print(f"✓ Chain works!\\n\\nTest answer:\\n{answer}")
"""))

# ─── GIT CHECKPOINT 2 ────────────────────────────────────────────────────────
cells.append(md("""
---

## 📌 Git Checkpoint 2 — Core Pipeline Working

**The hardest part is done — commit it.**  
A working chain is your v1 baseline. Every improvement after this point
can be measured against it.

**Fill in `NOTES` with one of your test answers, then run the cell.**
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# GIT CHECKPOINT 2
# ════════════════════════════════════════════════════════════════

NOTES = \"\"\"
Exercises: 4 (embeddings + vector store), 5 (RAG chain)
Status: ✓ / ⚠ / ✗   ← delete two

Embedding model : text-embedding-3-small
LLM             : gpt-4o-mini  temperature=0
Chunks indexed  : ???   ← fill in vector_store._collection.count()

Sample test answer:
  Q: What is RAG?
  A: ???   ← paste your answer here

Notes:
-
-
\"\"\"

import subprocess
staged = subprocess.run("git diff --cached --name-only",
                        shell=True, capture_output=True, text=True).stdout
assert ".env" not in staged, "🚨  .env is staged! Run: git reset HEAD .env"
print("✓ .env safety check passed")

subprocess.run("git add rag_eval_workflow.ipynb", shell=True)
subprocess.run(["git", "commit", "-m",
    f"[RAG-2] Vector store + RAG chain working\\n\\n{NOTES.strip()}"])
print("\\n✓ Committed!  Push when ready: git push origin main")
"""))

# ─── EXERCISE 6 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 6 — Test Your Pipeline with Real Queries

### Concept: Manual Testing Before Formal Evaluation

Before running automated metrics, probe the pipeline yourself. You are looking for:

| Test type | What to ask | What to look for |
|-----------|------------|-----------------|
| **In-scope fact** | Something clearly in the article | A specific, grounded answer |
| **Conceptual** | "What is the difference between X and Y?" | Multi-sentence explanation |
| **Out-of-scope** | Something NOT in the article | Should say "I don't know" |
| **Edge case** | Vague or ambiguous question | How does it handle uncertainty? |

**Instructions:**
1. Write at least **6 questions** in `my_questions` — cover all 4 test types above
2. Loop through them: call `rag_chain.invoke(q)` and print each answer
3. Observe: does the model ever hallucinate? Does it correctly say "I don't know"?

**Expected output:** 6+ question/answer pairs printed clearly.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 6 — Interactive Testing
# ════════════════════════════════════════════════════════════════

# ── YOUR CODE HERE ────────────────────────────────────────────
# Write at least 6 questions — include at least one that is NOT
# answerable from the article (to test the "I don't know" fallback)

my_questions = [
    # "...",   ← in-scope: simple fact
    # "...",   ← in-scope: conceptual
    # "...",   ← in-scope: comparison (e.g. RAG vs fine-tuning)
    # "...",   ← out-of-scope: something NOT in the article
    # "...",   ← your choice
    # "...",   ← your choice
]

# Loop through your questions and print each answer
# for q in my_questions:
#     answer = rag_chain.invoke(q)
#     print(f"Q: {q}")
#     print(f"A: {answer}")
#     print("-" * 60)


# ── SELF CHECK ───────────────────────────────────────────────
assert len(my_questions) >= 6, \\
    f"✗ Write at least 6 questions — you have {len(my_questions)}"
print(f"✓ Tested {len(my_questions)} questions.")
print("\\nReflection — answer these before moving on:")
print("  1. Did the model ever include information NOT in the article?")
print("  2. Did it correctly refuse off-topic questions?")
print("  3. Which answer impressed you most?")
"""))

# ─── GIT CHECKPOINT 3 ────────────────────────────────────────────────────────
cells.append(md("""
---

## 📌 Git Checkpoint 3 — Manual Testing Complete

**What to note:** Your test questions reveal your intuition about the system.
Writing them down before formal evaluation is valuable for your portfolio.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# GIT CHECKPOINT 3
# ════════════════════════════════════════════════════════════════

NOTES = f\"\"\"
Exercises: 6 (interactive testing)
Status: ✓ / ⚠ / ✗   ← delete two

Questions tested   : {len(my_questions)}
Hallucination seen : yes / no
"I don't know" worked: yes / no

Most interesting answer:
  Q: ???
  A: ???

Notes:
-
-
\"\"\"

import subprocess
staged = subprocess.run("git diff --cached --name-only",
                        shell=True, capture_output=True, text=True).stdout
assert ".env" not in staged, "🚨  .env is staged! Run: git reset HEAD .env"
print("✓ .env safety check passed")

subprocess.run("git add rag_eval_workflow.ipynb", shell=True)
subprocess.run(["git", "commit", "-m",
    f"[RAG-3] Manual query testing — {len(my_questions)} questions\\n\\n{NOTES.strip()}"])
print("✓ Committed!  Push when ready: git push origin main")
"""))

# ─── EXERCISE 7 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 7 — Build a Golden Evaluation Dataset

### Concept: What Is a Golden Dataset?

A golden dataset is your **ground truth** — questions with known correct answers
that *you* write before seeing the model's output. It is the foundation of
objective evaluation.

Each row has 4 fields:

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | A realistic user question |
| `answer` | `str` | The answer your RAG chain generated |
| `contexts` | `list[str]` | Raw chunk text retrieved (not the Document objects) |
| `ground_truth` | `str` | Your handwritten reference answer |

### Rules for writing good questions

- Write your `ground_truths` **before** running the model — don't peek
- Cover different sections of the article (not all from the introduction)
- Include simple lookups, conceptual questions, and at least one multi-hop question
- Write **at least 10** — fewer gives statistically unreliable metric averages

### Capturing contexts correctly

RAGAS needs the **raw text strings** from the retrieved chunks, not Document objects:
```python
docs = retriever.invoke(question)
context_texts = [doc.page_content for doc in docs]   # list[str]
```

### Building the Dataset

```python
eval_dataset = Dataset.from_dict({
    "question":    questions,           # list[str]
    "answer":      generated_answers,   # list[str]
    "contexts":    retrieved_contexts,  # list[list[str]]
    "ground_truth": ground_truths,      # list[str]
})
```

**Instructions:**
1. Write 10 questions in `questions` and matching reference answers in `ground_truths`
2. Loop through questions: retrieve chunks, generate answers, append to `generated_answers` and `retrieved_contexts`
3. Build `eval_dataset` with `Dataset.from_dict(...)`

**Expected output:** `Dataset({features: ['question','answer','contexts','ground_truth'], num_rows: 10})`
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 7 — Build the Golden Evaluation Dataset
# ════════════════════════════════════════════════════════════════

# ── YOUR CODE HERE — Part A: Write your Q&A pairs ─────────────
# Write 10 questions — read the Wikipedia article first!
questions = [
    # "...",   ← Q1
    # "...",   ← Q2
    # "...",   ← Q3
    # "...",   ← Q4
    # "...",   ← Q5
    # "...",   ← Q6
    # "...",   ← Q7
    # "...",   ← Q8
    # "...",   ← Q9
    # "...",   ← Q10
]

# Write reference answers — do this BEFORE running the model below
# Use the Wikipedia article as your source, not the model's output
ground_truths = [
    # "...",   ← GT1
    # "...",   ← GT2
    # "...",   ← GT3
    # "...",   ← GT4
    # "...",   ← GT5
    # "...",   ← GT6
    # "...",   ← GT7
    # "...",   ← GT8
    # "...",   ← GT9
    # "...",   ← GT10
]


# ── YOUR CODE HERE — Part B: Collect pipeline outputs ─────────
generated_answers  = []
retrieved_contexts = []   # must be list[list[str]] — raw text, not Document objects

# Loop through questions and for each one:
#   1. docs = retriever.invoke(q)
#   2. context_texts = [doc.page_content for doc in docs]
#   3. answer = rag_chain.invoke(q)
#   4. append answer and context_texts to their lists
#   5. print a progress line: print(f"  [{i}/{len(questions)}] {q[:50]}...")
#
# for i, q in enumerate(questions, 1):
#     ...


# ── YOUR CODE HERE — Part C: Build the Dataset ────────────────
# eval_dataset = Dataset.from_dict({
#     "question":    questions,
#     "answer":      generated_answers,
#     "contexts":    retrieved_contexts,
#     "ground_truth": ground_truths,
# })
eval_dataset = None  # replace this line


# ── SELF CHECK ───────────────────────────────────────────────
assert len(questions)    >= 10, f"✗ Need 10+ questions, you have {len(questions)}"
assert len(questions)    == len(ground_truths), \\
    "✗ questions and ground_truths must be the same length"
assert len(generated_answers)  == len(questions), \\
    "✗ Run the collection loop first to fill generated_answers"
assert len(retrieved_contexts) == len(questions), \\
    "✗ Run the collection loop first to fill retrieved_contexts"
assert eval_dataset is not None, "✗ Build eval_dataset with Dataset.from_dict(...)"
assert set(eval_dataset.column_names) >= {"question","answer","contexts","ground_truth"}, \\
    f"✗ Dataset missing columns — got {eval_dataset.column_names}"

print(f"✓ Golden dataset ready: {len(eval_dataset)} rows")
print(f"  Columns: {eval_dataset.column_names}")
print(f"\\nPreview — Row 1:")
print(f"  Q  : {eval_dataset[0]['question']}")
print(f"  GT : {eval_dataset[0]['ground_truth'][:100]}")
print(f"  A  : {eval_dataset[0]['answer'][:100]}...")
print(f"  Ctx: {len(eval_dataset[0]['contexts'])} chunks retrieved")
"""))

# ─── GIT CHECKPOINT 4 ────────────────────────────────────────────────────────
cells.append(md("""
---

## 📌 Git Checkpoint 4 — Golden Dataset Written

**What to note:** Ground truths are original intellectual work.
Committing them separately proves you wrote them before seeing the model's output.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# GIT CHECKPOINT 4
# ════════════════════════════════════════════════════════════════

NOTES = f\"\"\"
Exercises: 7 (golden evaluation dataset)
Status: ✓ / ⚠ / ✗   ← delete two

Questions written : {len(questions) if 'questions' in dir() else '???'}
Ground truth source: Wikipedia article (written before running the model)

Topics covered:
  -
  -
  -

Notes (what made writing ground truths harder than expected):
-
\"\"\"

import subprocess
staged = subprocess.run("git diff --cached --name-only",
                        shell=True, capture_output=True, text=True).stdout
assert ".env" not in staged, "🚨  .env is staged! Run: git reset HEAD .env"
print("✓ .env safety check passed")

subprocess.run("git add rag_eval_workflow.ipynb", shell=True)
q_count = len(questions) if 'questions' in dir() else '?'
subprocess.run(["git", "commit", "-m",
    f"[RAG-4] Golden dataset — {q_count} handwritten Q&A pairs\\n\\n{NOTES.strip()}"])
print("✓ Committed!  Push when ready: git push origin main")
"""))

# ─── EXERCISE 8 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 8 — Run RAGAS Evaluation

### Concept: The 4 RAGAS Metrics

RAGAS uses the LLM itself as a judge to score your pipeline on 4 dimensions:

| Metric | Question it answers | Failure = |
|--------|--------------------|----|
| `faithfulness` | Is every claim in the answer supported by the retrieved context? | Hallucination |
| `answer_relevancy` | Does the answer actually address what was asked? | Vague / off-topic |
| `context_recall` | Does the retrieved context contain the ground truth information? | Wrong chunks retrieved |
| `context_precision` | Are most retrieved chunks relevant (low noise)? | Too much junk retrieved |

**Score scale: 0.0 (worst) → 1.0 (best)**

A strong production RAG system typically scores > 0.80 across all metrics.

### Running evaluation

```python
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
results_df = results.to_pandas()
```

> ⚠️ RAGAS makes LLM API calls internally to score each row.  
> With 10 questions × 4 metrics this takes **1–3 minutes** and costs ~**$0.05**.

**Instructions:**
1. Create a list `metrics` containing all 4 metric objects
2. Call `evaluate(dataset=eval_dataset, metrics=metrics)` — store as `results`
3. Convert to a DataFrame: `results_df = results.to_pandas()`
4. Display `results_df`

**Expected output:** A table with columns `question`, `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 8 — Run RAGAS Evaluation
# ════════════════════════════════════════════════════════════════

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

# ── YOUR CODE HERE ────────────────────────────────────────────
# Step 1: Create the metrics list — include all 4 metric objects
#   metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
metrics = None  # replace this line

# Step 2: Run the evaluation — this calls the LLM internally as a judge
#   results = evaluate(dataset=eval_dataset, metrics=metrics)
print("Running RAGAS evaluation — takes 1–3 minutes...")
results = None  # replace this line

# Step 3: Convert to a pandas DataFrame
#   results_df = results.to_pandas()
results_df = None  # replace this line

# Step 4: Display the DataFrame
# display(results_df[["question"] + METRIC_COLS])


# ── SELF CHECK ───────────────────────────────────────────────
assert metrics is not None and len(metrics) == 4, \\
    "✗ Create a list of exactly 4 RAGAS metric objects"
assert results    is not None, "✗ Call evaluate() and store the result"
assert results_df is not None, "✗ Call results.to_pandas() and store as results_df"
assert all(col in results_df.columns for col in METRIC_COLS), \\
    f"✗ DataFrame missing metric columns. Got: {list(results_df.columns)}"
print("\\n✓ Evaluation complete!")
print(f"  Rows: {len(results_df)}  |  Columns: {list(results_df.columns)}")
"""))

# ─── EXERCISE 9 ───────────────────────────────────────────────────────────────
cells.append(md("""
---

## Exercise 9 — Analyze & Visualize Results

### Concept: Reading Your Scores

Print the aggregate (mean) score for each metric, then visualize per-question
scores to find patterns in failures.

**Score interpretation guide:**

| Score range | Meaning | Fix |
|------------|---------|-----|
| `faithfulness` < 0.7 | LLM is hallucinating — ignoring context | Tighten system prompt, lower temperature |
| `answer_relevancy` < 0.7 | Answers are vague or off-topic | Improve prompt clarity |
| `context_recall` < 0.7 | Retriever misses key chunks | Reduce chunk size, increase `k` |
| `context_precision` < 0.7 | Too much irrelevant context retrieved | Increase threshold, use MMR retrieval |

### Visualizations to build

1. **Bar chart** — one bar per metric showing the mean score (axes[0])
2. **Heatmap** — rows = questions (Q1–Q10), columns = 4 metrics, color = score (axes[1])

**Instructions:**
1. Print a formatted table of mean scores for all 4 metrics
2. Build the bar chart on `axes[0]` — add a dashed "target" line at 0.8
3. Build the heatmap on `axes[1]` with `seaborn.heatmap` using `cmap="RdYlGn"`
4. Identify the 3 worst questions using `nsmallest(3, "avg_score")`
5. Export `results_df` to `./rag_eval_results.csv`

**Expected output:** Score table, two-panel chart, worst-question list, CSV saved.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# EXERCISE 9 — Analyze & Visualize Results
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import seaborn as sns

# ── YOUR CODE HERE — Part A: Print aggregate scores ───────────
# For each metric in METRIC_COLS print:
#   metric name, mean score, a bar of █ characters (int(score*20) filled)
#
# Example output:
#   faithfulness          0.823  [████████████████░░░░]
#
# for metric in METRIC_COLS:
#     score = results_df[metric].mean()
#     bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
#     print(f"  {metric:<25} {score:.3f}  [{bar}]")


# ── YOUR CODE HERE — Part B: Two-panel chart ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# axes[0]: Bar chart of mean scores
#   - x-axis: METRIC_COLS
#   - y-axis: mean scores, range 0–1
#   - add a dashed horizontal line at 0.8 labelled "Target (0.8)"
#   - print the score value above each bar


# axes[1]: Heatmap — per-question scores
#   heatmap_data = results_df[METRIC_COLS].copy()
#   heatmap_data.index = [f"Q{i+1}" for i in range(len(heatmap_data))]
#   sns.heatmap(heatmap_data, ax=axes[1], annot=True, fmt=".2f",
#               cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5)


plt.tight_layout()
plt.show()


# ── YOUR CODE HERE — Part C: Worst questions ─────────────────
# Compute avg_score = mean of the 4 metric columns per row
# Print the 3 questions with the lowest avg_score
#
# results_df["avg_score"] = results_df[METRIC_COLS].mean(axis=1)
# worst = results_df.nsmallest(3, "avg_score")
# for _, row in worst.iterrows():
#     print(f"  Q: {row['question']}")
#     print(f"     avg={row['avg_score']:.3f}  " + ...)


# ── YOUR CODE HERE — Part D: Export ──────────────────────────
# results_df.to_csv("./rag_eval_results.csv", index=False)
# print("✓ Results saved to rag_eval_results.csv")


# ── SELF CHECK ───────────────────────────────────────────────
import os
assert os.path.exists("./rag_eval_results.csv"), \\
    "✗ CSV not found — did you call results_df.to_csv('./rag_eval_results.csv', index=False)?"
assert "avg_score" in results_df.columns, \\
    "✗ avg_score column missing — compute it with results_df[METRIC_COLS].mean(axis=1)"
print("✓ CSV saved.  Run `make check-eval` to validate scores.")
"""))

# ─── GIT CHECKPOINT 5 ────────────────────────────────────────────────────────
cells.append(md("""
---

## 📌 Git Checkpoint 5 — Project Complete

**This is your final commit.** It auto-fills your RAGAS scores into the commit
message so your portfolio commit history shows the actual numbers.
"""))

cells.append(code("""
# ════════════════════════════════════════════════════════════════
# GIT CHECKPOINT 5 — FINAL
# ════════════════════════════════════════════════════════════════

# Auto-fill scores from results_df
try:
    scores_str = "\\n".join(
        f"  {m:<22}: {results_df[m].mean():.3f}" for m in METRIC_COLS
    )
    weakest = min(METRIC_COLS, key=lambda m: results_df[m].mean())
except Exception:
    scores_str = "  (run Exercise 8 first)"
    weakest    = "unknown"

NOTES = f\"\"\"
Exercises: 8 (RAGAS evaluation), 9 (analysis + visualization)
Status: ✓ / ⚠ / ✗   ← delete two

RAGAS Scores (mean across {len(questions) if 'questions' in dir() else '?'} questions):
{scores_str}

Weakest metric  : {weakest}
Root cause      : ???   ← explain why this metric is low
Planned fix     : ???   ← what would you change first

Key takeaways:
-
-
\"\"\"

import subprocess
staged = subprocess.run("git diff --cached --name-only",
                        shell=True, capture_output=True, text=True).stdout
assert ".env" not in staged, "🚨  .env is staged! Run: git reset HEAD .env"
print("✓ .env safety check passed")

subprocess.run("git add rag_eval_workflow.ipynb rag_eval_results.csv 2>/dev/null || true",
               shell=True)
subprocess.run(["git", "commit", "-m",
    f"[RAG-5] FINAL — evaluation complete\\n\\n{NOTES.strip()}"])

# Push everything
print("\\nPushing all checkpoints to GitHub...")
result = subprocess.run("git push origin main",
                        shell=True, capture_output=True, text=True)
print(result.stdout or result.stderr)
print("\\n" + "═"*50)
print("  PROJECT COMPLETE")
print("  https://github.com/Dashakid/RAG-101")
print("  Run: git log --oneline  to see your history")
print("═"*50)
"""))

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
cells.append(md("""
---

## 🎓 You're Done

### What You Built

```
Wikipedia article  ──►  WebBaseLoader  ──►  RecursiveCharacterTextSplitter
                                                        │
                              OpenAIEmbeddings  +  ChromaDB
                                                        │
User Question  ──────────────────────────►  Retriever (k=4)
                                                        │
                        ChatPromptTemplate  +  gpt-4o-mini
                                                        │
                                    Answer  ──►  RAGAS  ──►  CSV
```

---

### Reflection Questions

Write your answers in a new markdown cell below each one:

1. **What was your lowest-scoring metric?**  
   What does that tell you about your pipeline's specific weakness?

2. **Look at your 3 worst questions.**  
   Is there a pattern — all from the same section, all a certain type?

3. **What would you change first** to improve your lowest score?  
   (chunk size / k / prompt / LLM choice)

---

### Next Challenges

| Challenge | What to change |
|-----------|---------------|
| Improve context recall | Increase `k` from 4 to 6 and re-evaluate |
| Reduce retrieval noise | Switch to `search_type="mmr"` |
| Try different documents | Load a PDF with `PyPDFLoader` |
| Tune chunk size | Try 256 and 1024 — compare scores |
| Add memory | Wrap chain with `ConversationBufferMemory` |
| Deploy it | Wrap `rag_chain.invoke()` in a FastAPI route |
"""))

# ─── WRITE FILE ───────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "lewagon", "language": "python", "name": "lewagon"},
        "language_info": {"name": "python", "version": "3.12.9"}
    },
    "cells": cells
}

with open("rag_eval_workflow.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"✓ Notebook written: {len(cells)} cells")
print(f"  Code cells    : {sum(1 for c in cells if c['cell_type']=='code')}")
print(f"  Markdown cells: {sum(1 for c in cells if c['cell_type']=='markdown')}")
