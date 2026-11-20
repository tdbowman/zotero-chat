# Zotero Chat

**Chat with your Zotero library: the papers, your notes and your PDF highlights,
with answers that cite the source and page. Everything runs on your own computer.**

Zotero Chat is a local retrieval-augmented generation (RAG) app. It reads your
Zotero libraries and builds a separate search index for each one: My Library and
each group library. You then ask questions in a chat window, optionally limited to
one collection. Answers are written by a local LLM using only passages from your
library, and cite them like `[Smith et al. 2019, p. 4]`. Each source links back to
the item in Zotero, or opens the PDF at that page.

No cloud services are involved. Models run in [Ollama](https://ollama.com),
indexes are stored on disk, and your library never leaves your machine.

```
 ┌──────────────┐     build_index.py      ┌──────────────────────┐        app.py         ┌──────────────┐
 │    Zotero    │ ──────────────────────▶ │  indexes/my-library  │ ◀──────────────────── │  Streamlit   │
 │ (local API / │  items, PDFs, notes,    │  indexes/group-1234  │  hybrid search,       │  chat in     │
 │  zotero.db)  │  highlights → chunks    │  ...  (LanceDB)      │  rerank, cited answer │  your browser│
 └──────────────┘  → embeddings           └──────────────────────┘                       └──────────────┘
```

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Building your indexes](#building-your-indexes)
- [Using the app](#using-the-app)
- [Keeping indexes up to date](#keeping-indexes-up-to-date)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Privacy and copyright](#privacy-and-copyright)
- [Limitations](#limitations)
- [License](#license)

## Features

- **One index per Zotero library.** Pick My Library or any group library from a
  dropdown; each keeps its own conversation.
- **Collection filter.** Search within one collection, including its subcollections.
- **Everything you keep in Zotero:**
  - the item's details and abstract
  - the **full text** of PDFs, saved web pages and text files, with page numbers
  - **your notes**
  - **your PDF highlights and comments**
- **Cited answers.** Every claim cites `[Author Year, p. N]`. The Sources panel
  shows each passage used, with **Show in Zotero**, **Open PDF at page N** and
  DOI links.
- **Honest when it doesn't know.** If nothing in the library is relevant enough,
  it says so instead of guessing.
- **Follow-up questions.** Questions like "What method did they use?" are
  rewritten into a standalone search using the conversation so far.
- **Fast incremental updates.** Only items that changed are re-indexed, and
  extracted PDF text is cached. When reading through Zotero's local API,
  libraries that haven't changed are skipped entirely.
- **Works with Zotero open or closed.** It uses Zotero's local API when Zotero is
  running, and otherwise reads a snapshot copy of its database. Nothing in Zotero
  is ever modified.

## Requirements

| | |
|---|---|
| **Computer** | Developed and tested on macOS with Apple Silicon (16 GB+ memory recommended). It should also work on Linux and Windows; the reranker uses an NVIDIA GPU if present, otherwise the CPU (slower). |
| **Zotero** | [Zotero 7](https://www.zotero.org/download/) or later, with your library synced to this computer. |
| **Ollama** | [ollama.com](https://ollama.com), to run the LLM and the embedding model locally. |
| **Python** | 3.12 or 3.13 (tested with 3.13). |
| **Disk** | About 8 GB for the models. Indexes take roughly 7 KB per passage: a library of 2,700 items with ~1,600 PDFs produced ~69,000 passages and ~490 MB, plus ~90 MB of cached PDF text. |

## Setup

### 1. Zotero

1. Install Zotero 7 and sign in under **Settings → Sync**, so your libraries,
   including any group libraries, are on this computer.
2. Make sure the files themselves are downloaded, not just the item records. In
   **Settings → Sync → File Syncing**, set *Download files* to **at sync time**,
   for My Library and for groups. With "as needed", PDFs you haven't opened aren't
   on disk and can't be indexed.
3. Turn on the local API: **Settings → Advanced → "Allow other applications on
   this computer to communicate with Zotero"**. You can check it in a browser:
   <http://localhost:23119/api/users/0/items?limit=1> should return JSON.
   This is optional, but recommended (see [How it works](#how-it-works)).

### 2. Ollama and the models

Install Ollama from [ollama.com/download](https://ollama.com/download), or on macOS
run `brew install ollama && brew services start ollama`. Then download the two models:

```bash
ollama pull embeddinggemma          # embeddings, ~0.6 GB
ollama pull gemma4:e4b-it-q4_K_M    # answers, ~6.6 GB
```

Any Ollama chat model can write the answers: set `LLM_MODEL` in `.env`. On a
machine with less memory, a smaller model such as `llama3.2` (~2 GB) works,
though its answers are less careful.

### 3. This app

```bash
git clone https://github.com/tdbowman/zotero-chat.git
cd zotero-chat

python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt    # includes PyTorch; takes a few minutes

cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and check `ZOTERO_DATA_DIR`: it must be the folder that contains
`zotero.sqlite` and `storage/`. The default `~/Zotero` is right for most people.
In Zotero, see **Settings → Advanced → Files and Folders → Data Directory
Location**. Every other setting has a working default; see
[Configuration](#configuration).

The reranker model (~0.4 GB) downloads from Hugging Face the first time you ask a
question.

## Building your indexes

List your libraries:

```bash
python build_index.py --list
```

```
slug                  items  indexed  name
my-library             2678           My Library
group-1234567           528           Lab Reading Group
group-7654321            85           Course Readings
...
```

Build one library, several, or all of them. Libraries can be named by name, group
ID or slug:

```bash
python build_index.py --library "My Library"
python build_index.py --library 1234567 --library "Course Readings"
python build_index.py --all
```

The first build of a large library takes a while: it extracts the text of every
PDF and embeds every passage. On an Apple M4 Mac, a library of 2,600 items and
1,600 PDFs (~70,000 passages) took about 50 minutes; embedding runs at roughly
25 passages per second. Later runs only process what
changed (see [Keeping indexes up to date](#keeping-indexes-up-to-date)).

Each build reports what it found:

```
My Library: 2678 items; reading attachments (cached text is reused)…
  68821 chunks {'reference': 2689, 'fulltext': 65705, 'note': 427}; tokens median 342, max 477
  attachments: {'empty': 19, 'missing': 28, 'no-file': 60, 'ok': 1614}
```

| Attachment status | Meaning |
|---|---|
| `ok` | Text extracted. |
| `missing` | Zotero knows the file, but it isn't on disk: not downloaded yet, or never uploaded to Zotero storage by whoever added it. |
| `no-file` | A link to a web page, with no stored copy. |
| `empty` | No extractable text, usually a scanned PDF without a text layer. Zotero's own OCR'd text (`.zotero-ft-cache`) is used when available. |
| `error` | The file couldn't be read (corrupt, encrypted…). |

All `build_index.py` options:

| Option | |
|---|---|
| `--list` | List libraries, item counts, and which are indexed. |
| `--library X` | Build or update this library (name, group ID or slug). Repeatable. |
| `--all` | Build or update every library that has items. |
| `--rebuild` | Delete the index(es) and start from scratch. Only needed after changing the embedding model, `EMBED_DOC_PREFIX` or `CHUNK_TOKENS`. |
| `--dry-run` | Read and chunk, print statistics, don't embed or write anything. |
| `--keep-references` | Also index papers' reference lists, which are skipped by default (see [How it works](#how-it-works)). |

## Using the app

```bash
./start-app.sh                     # macOS / Linux, http://localhost:8502
# or, on any system:
streamlit run app.py --server.port 8502
```

Open <http://localhost:8502>. In the sidebar:

- **Library**: which index to chat with. Each library keeps its own
  conversation; **New conversation** clears the current one.
- **Collection**: limit the search to one collection and its subcollections.
- Index statistics, and a warning if attachment files were missing when the index
  was built.

The first question after choosing a library takes a few extra seconds while its
index and the models load. After that, answers start in about 5 seconds and finish
in about 15 on an M4 Mac.

Questions that work well:

- *What does my library say about Mendeley readership as an early indicator of citations?*
- *Which papers discuss selfies and self-presentation?*
- *How do the sources define "altmetrics"?* … then *Which of them are critical of it?*
- *Summarize my notes on research data management.*
- *What did I highlight about trust in online communities?*

Under each answer, **Sources** lists the passages it was based on, with their
relevance scores. **Show in Zotero** and **Open PDF at p. N** use `zotero://`
links, so they open the Zotero desktop app.

Engine only, from the terminal:

```bash
python zotero_rag.py my-library "What does my library say about open peer review?"
```

## Keeping indexes up to date

Re-run the builder after you add, edit or delete items, or after new files sync:

```bash
python build_index.py --all
```

Updates are incremental:

- Every item's passages carry a fingerprint of their content. Only items whose
  fingerprint changed are re-embedded, and items deleted from Zotero (or moved to
  the trash) are removed.
- Extracted attachment text is cached in `cache/text/`, keyed by each file's
  size and modification time. PDFs are only re-read when the file changes.
- Through the local API, each library reports a version number that changes
  whenever anything in it changes. If it hasn't changed since the last build, and
  no files were missing then, the library is skipped in under a second.

To run this automatically, schedule it with `cron` (Linux), a LaunchAgent
(macOS) or Task Scheduler (Windows). The app picks up a rebuilt index the next
time you choose that library.

## Configuration

All settings live in `.env`. [`.env.example`](.env.example) lists every one with an
explanation; the most useful are:

| Setting | Default | |
|---|---|---|
| `ZOTERO_DATA_DIR` | `~/Zotero` | Folder with `zotero.sqlite` and `storage/`. |
| `ZOTERO_SOURCE` | `auto` | `auto`, `api` or `sqlite`; see [How it works](#how-it-works). |
| `LLM_MODEL` | `gemma4:e4b-it-q4_K_M` | Ollama model that writes answers. |
| `EMBEDDING_MODEL` | `embeddinggemma` | Ollama embedding model (rebuild indexes after changing). |
| `RERANKER_MODEL` | `mixedbread-ai/mxbai-rerank-base-v1` | Cross-encoder; empty = no reranking. |
| `RERANK_TOP_N` | `6` | Passages given to the LLM. |
| `RERANK_MIN_SCORE` | `0.1` | Relevance cutoff (0–1); raise it for fewer, more on-topic passages. |
| `MAX_CHUNKS_PER_ITEM` | `3` | Cap per item, so one long paper can't fill every slot. |
| `NUM_QUERY_VARIATIONS` | `3` | Search with the question plus 2 LLM rephrasings; `1` is faster. |
| `CHUNK_TOKENS` | `350` | Passage size (rebuild indexes after changing). |

## How it works

### Reading Zotero (`zotero_reader.py`)

There are two interchangeable sources, which produce identical item records:

- **Local API** (`ApiSource`, preferred). Zotero 7's built-in read-only API at
  `localhost:23119/api`. This is Zotero's supported interface, so it keeps working
  when Zotero changes its internal database. It also gives each library a version
  number, which allows skipping unchanged libraries. It needs Zotero to be running.
- **Database snapshot** (`SqliteSource`, fallback). Copies `zotero.sqlite`, plus
  its write-ahead log, to a temporary folder and reads the copy. Zotero locks the
  live file while it runs, and a copy is never written back. This works with
  Zotero closed, but depends on Zotero's internal tables.

With `ZOTERO_SOURCE=auto`, the API is used when it answers, and the snapshot
otherwise. Both give identical passages (checked item by item on a 10,000-item
set of libraries), so switching never causes re-embedding.

Attachment files are read directly from `storage/` (or their linked location),
because the local API doesn't serve full text. PDFs are read page by page with
[PyMuPDF](https://pymupdf.readthedocs.io), keeping the printed page labels, and
HTML snapshots are read with BeautifulSoup. For other types (Word files, for
example), Zotero's own extracted text is used when present.

### Building an index (`build_index.py`)

Each item becomes several passages ("chunks"). Every chunk starts with a label line,
so it makes sense on its own to the search, the reranker and the LLM. The LLM
copies that label as its citation:

| Chunk type | Content | Label |
|---|---|---|
| `reference` | type, authors, year, venue, DOI, collections, tags, abstract | `[Smith et al. 2019] Title` |
| `fulltext` | attachment text, ~350 tokens per chunk | `[Smith et al. 2019, p. 4] Title` |
| `note` | your notes | `[Smith et al. 2019] Title (your note)` |
| `annotation` | highlights and comments, in reading order | `[Smith et al. 2019] Title (your highlights and comments)` |

Lines of mis-encoded PDF text (fonts without a Unicode mapping, which come out as
`ŵŽŶƚŚ`-style symbols) are dropped. Full text is split into sentence-aligned pieces
within each page. Short pieces are
merged across page breaks, and a merged chunk is labelled with the page it starts on.
A paper's **References section is skipped**: dense lists of other authors and titles
match many questions while answering none. Only a "References" / "Bibliography" /
"Works Cited" heading in the second half of the document counts.

No chunk exceeds 480 tokens, because the reranker reads only the first 512. Long
author lists are cut to 10 names in the reference card. Chunks are embedded with
EmbeddingGemma through Ollama and stored in
[LanceDB](https://lancedb.com), one folder per library: `indexes/my-library/`,
`indexes/group-<groupID>/`. Each folder's `BUILD_INFO.json` records the settings
used, statistics and the library version.

### Answering a question (`zotero_rag.py`)

1. **Follow-ups** are rewritten as standalone questions by the LLM, using the last
   few exchanges.
2. **Library-style framing is removed for search.** "What does my library say
   about X?" or "Which papers discuss X?" is searched as just *X*. The framing
   alone pushed the reranker's score for a clearly relevant abstract from 0.35 to
   0.01. The LLM still sees the full question.
3. **Hybrid search.** Vector similarity (meaning) and BM25 (exact words: names,
   acronyms, terms) run for the question and two LLM-written rephrasings. The
   results are fused by reciprocal rank into 25 candidates, optionally within one
   collection.
4. **Reranking.** A cross-encoder (`mxbai-rerank-base-v1`) reads the question
   with each candidate and scores its relevance from 0 to 1. Duplicates are
   dropped, including the same PDF saved under two items, plus anything below
   `RERANK_MIN_SCORE`. At most 3 chunks per item are kept, and the best 6 go
   forward. If nothing is left, the app says the library doesn't cover the
   question.
5. **Answer.** The local LLM streams an answer that uses only those passages and
   cites their labels. It is told to point out where sources disagree, and to call
   your own notes "your notes".

## Project layout

| File | |
|---|---|
| [`zotero_reader.py`](zotero_reader.py) | Reads Zotero: `ApiSource` (local API) and `SqliteSource` (database snapshot) return libraries and items with metadata, collections, tags, notes, annotations and attachment paths. Also handles attachment text extraction (cached) and `zotero://` links. |
| [`build_index.py`](build_index.py) | Command-line index builder: chunking, fingerprints, incremental embedding into LanceDB. |
| [`zotero_rag.py`](zotero_rag.py) | The engine: `LibraryIndex` (hybrid retrieval, collection filter, reranking, follow-up rewriting, streaming answers) and `list_indexes()`. |
| [`app.py`](app.py) | Streamlit chat UI: library and collection pickers, per-library conversations, sources panel. |
| [`start-app.sh`](start-app.sh) | Starts the app on `localhost:8502` after checking Ollama. |
| [`.env.example`](.env.example) | Every setting, documented. Copy to `.env`. |
| `indexes/`, `cache/`, `logs/` | Created when you build. Personal data, ignored by git. |

## Troubleshooting

**"Zotero local API not reachable … using a database snapshot"**
: Expected when Zotero is closed. If Zotero is open, enable the local API
  (Setup step 1.3), or set `ZOTERO_SOURCE=sqlite`.

**"No Zotero database at …"**
: Set `ZOTERO_DATA_DIR` in `.env` to your Zotero data directory.

**Many attachments reported `missing`**
: The files aren't in `storage/`. Set Zotero's *Download files* to "at sync time"
  and let it sync, then re-run the build: only the new files are processed. If
  files stay missing, they were probably never uploaded to Zotero storage from the
  computer that added them (the item exists, the file doesn't).

**"Ollama isn't running" / connection refused on port 11434**
: Start Ollama (`brew services start ollama`, or open the Ollama app), then check
  that `ollama list` shows both models.

**"Index … was built with … ; .env says …"**
: The embedding model or `EMBED_DOC_PREFIX` changed since the index was built.
  Vectors from different models can't be mixed: run
  `python build_index.py --all --rebuild`.

**Answers say the library has nothing relevant, but it does**
: Try naming the topic more specifically, check the collection filter, or lower
  `RERANK_MIN_SCORE` (e.g. to `0.05`). Scanned PDFs without a text layer can't be
  searched: run OCR on them (for example with `ocrmypdf`) and rebuild.

**The reranker download fails at the first question**
: The first question needs internet access to fetch the reranker from Hugging
  Face. Afterwards you can set `HF_HUB_OFFLINE=1`.

**Slow answers**
: Use a smaller `LLM_MODEL`, set `NUM_QUERY_VARIATIONS=1`, or lower `RERANK_TOP_N`.
  On computers without a GPU, the reranker runs on the CPU and adds a few seconds.

## Privacy and copyright

- Nothing is sent anywhere. The models run locally in Ollama. The only network
  access is the one-time model downloads, from Ollama and from Hugging Face for
  the reranker.
- `indexes/` and `cache/` contain the **full text of your PDFs**, plus your notes
  and highlights. They're in `.gitignore`: don't commit or share them, since most
  papers are copyrighted and your notes are yours.
- The app listens on `localhost` only. Don't expose it to a network without adding
  authentication: anyone who can reach it can query your library.

## Limitations

- Answers are only as good as the retrieved passages and the local model. Check
  important claims against the cited pages: the links are there for that.
- Scanned PDFs without a text layer contribute only their metadata (unless Zotero
  has OCR'd text for them).
- Questions that need the whole library at once ("How many papers did I add in
  2021?", "List every paper by Smith") aren't what retrieval does well: it finds
  the most relevant passages, not every match. Use Zotero's own search for those.
- Equations, tables and figures are read as plain text, at best.

## License

[MIT](LICENSE). Note that [PyMuPDF](https://pymupdf.readthedocs.io), which is used for
PDF text extraction, is licensed separately under the AGPL-3.0 (or a commercial
license). Check that it suits your use, especially if you distribute a modified
version.
