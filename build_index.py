#!/usr/bin/env python3
"""
Build a Search Index for a Zotero Library
-----------------------------------------
One LanceDB index per Zotero library, in indexes/<slug>/ ("my-library" or
"group-<groupID>"). The chat app lists every index found there.

Each item becomes several chunks, each starting with a header line such as
"[Smith et al. 2019, p. 4] Title" so it makes sense on its own to the
embedding model, BM25, the reranker and the LLM (which copies the bracketed
label as its citation):

  - reference:  title, authors, year, venue, DOI, collections, tags, abstract
  - fulltext:   PDF / snapshot text in ~350-token chunks with page numbers
                (the References section at the end of a paper is skipped)
  - note:       your Zotero notes
  - annotation: your PDF highlights and comments

Updates are incremental: every chunk records a fingerprint of its item, and
only items whose fingerprint changed (edited, new attachment synced, new
note...) are re-embedded; items removed from Zotero are dropped. Extracted
attachment text is cached (see zotero_reader.py), so re-running is quick.

Usage:
    python build_index.py --list                      # libraries and item counts
    python build_index.py --library "My Library"      # name, group ID or slug; repeatable
    python build_index.py --all                       # every library with items
    python build_index.py --library 245793 --rebuild  # start that index from scratch
    python build_index.py --library "My Library" --dry-run   # chunk statistics only
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

import zotero_reader as zr   # noqa: E402 (needs .env loaded for ZOTERO_DATA_DIR)

INDEX_ROOT = APP_DIR / os.getenv("INDEX_ROOT", "indexes")
TABLE = "zotero_chunks"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "title: none | text:")
OLLAMA_BASE_URL = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
# ~350 tokens holds a paragraph or two of research prose; the
# reranker reads at most 512 tokens, so chunks stay comfortably below that.
TARGET_TOKENS = int(os.getenv("CHUNK_TOKENS", "350"))
OVERLAP_TOKENS = 50
MAX_CHUNK_TOKENS = 480   # hard ceiling; the reranker reads at most 512 tokens
MIN_WORDS = 25            # fulltext pieces shorter than this are merged or dropped
EMBED_BATCH_ITEMS = 40    # items embedded and written per batch (progress is saved per batch)

# Every chunk carries exactly these metadata keys, all strings: LanceDB stores
# metadata as one struct column, so the schema must not vary between chunks.
META_KEYS = ("item_key", "attachment_key", "chunk_type", "title", "citation", "authors", "year",
             "item_type", "publication", "collections", "tags", "doi", "url", "page", "page_index",
             "library", "group_id", "fingerprint")

REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|reference list|references and notes)\s*$",
    re.I | re.M)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("BuildIndex")


# ------------------------------------------------------------------ chunking

class Chunker:
    """Token counting and sentence-aware splitting (LlamaIndex's tokenizer and splitter)."""

    def __init__(self):
        """Load LlamaIndex's tokenizer and a sentence splitter sized to TARGET_TOKENS."""
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.utils import get_tokenizer
        self._tok = get_tokenizer()
        self.splitter = SentenceSplitter(chunk_size=TARGET_TOKENS, chunk_overlap=OVERLAP_TOKENS)

    def tokens(self, text: str) -> int:
        """Number of tokens in `text` (the same tokenizer LlamaIndex uses for chunk sizes)."""
        return len(self._tok(text))

    def split(self, text: str):
        """Text in pieces of at most TARGET_TOKENS (whole if it already fits)."""
        return [text] if self.tokens(text) <= TARGET_TOKENS else self.splitter.split_text(text)


def drop_references(pages):
    """
    Cut a paper's text at its References heading.

    Reference lists are dense with other authors' names and titles, so they match
    many questions while answering none. Only a heading in the second half of the
    document counts, so a short "References" mention early on is ignored.
    """
    if len(pages) < 2:
        return pages
    start = len(pages) // 2
    for i in range(len(pages) - 1, start - 1, -1):
        m = REFERENCES_HEADING.search(pages[i]["text"])
        if m:
            kept = pages[:i]
            head = pages[i]["text"][:m.start()].strip()
            if head:
                kept.append({**pages[i], "text": head})
            return kept
    return pages


# Characters that show up when a PDF's fonts lack a proper Unicode mapping: control
# characters, and letters from Latin Extended-B / IPA / Coptic used in place of ASCII
# ("ŵŽŶƚŚ" for "month"). Real text in these ranges is rare; Cyrillic, Greek, CJK etc.
# are not affected.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_GARBLE = re.compile(r"[\u0180-\u02ff\u03e2-\u03ff]")


def drop_garbled(text: str) -> str:
    """
    Remove lines of mis-encoded PDF text.

    Such lines are unreadable for the LLM, match nothing useful, and take many
    tokens each (one 61-word garbled chunk measured 1,271 tokens). A line is
    dropped if it contains control characters, or if 30% or more of its
    non-space characters come from the ranges in _GARBLE.
    """
    kept = []
    for line in text.split("\n"):
        chars = [c for c in line if not c.isspace()]
        if _CONTROL.search(line):
            continue
        if chars and sum(1 for c in chars if _GARBLE.match(c)) >= 0.3 * len(chars):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def fulltext_pieces(pages, ch: Chunker):
    """
    Page-aware chunks: [(page_label, page_index, text)].

    Each page is split into sentence-aligned pieces; short pieces are merged with
    the next ones (across page breaks) until near the target size. A merged chunk
    is labelled with the page it starts on.
    """
    out, buf, buf_page = [], [], None
    buf_tokens = 0

    def flush():
        """Emit the buffered pieces as one chunk; a short leftover joins the previous chunk."""
        nonlocal buf, buf_page, buf_tokens
        text = "\n".join(buf).strip()
        if len(text.split()) >= MIN_WORDS:
            out.append((buf_page["page"], buf_page["index"], text))
        elif text and out:
            # A short leftover (e.g. the end of the last page) joins the previous chunk
            out[-1] = (out[-1][0], out[-1][1], f"{out[-1][2]}\n{text}")
        buf, buf_page, buf_tokens = [], None, 0

    for p in pages:
        for piece in ch.split(drop_garbled(p["text"])):
            n = ch.tokens(piece)
            if buf and buf_tokens + n > TARGET_TOKENS:
                flush()
            if not buf:
                buf_page = p
            buf.append(piece)
            buf_tokens += n
            if buf_tokens >= TARGET_TOKENS * 0.7:
                flush()
    if buf:
        flush()
    return out


MAX_CARD_AUTHORS = 10   # longer author lists (100+ on some papers) are cut to this in the card


def authors_text(item, limit: int = 0) -> str:
    """
    Authors as 'Last, First; Last, First' (editors etc. if the item has no authors).

    Args:
        item: An item record from zotero_reader.
        limit: Show at most this many names, then "et al. (N authors)"; 0 = all.
    """
    people = [c for c in item["creators"] if c["type"] == "author"] or item["creators"]
    names = [f"{c['last']}, {c['first']}".strip(", ") for c in people]
    if limit and len(names) > limit:
        return "; ".join(names[:limit]) + f"; et al. ({len(names)} authors)"
    return "; ".join(names)


def enforce_max_tokens(chunks, ch: Chunker):
    """
    Re-split any chunk that is still far over the target size.

    The reranker reads at most 512 tokens and silently ignores the rest, so a
    chunk above MAX_CHUNK_TOKENS is split again, each piece keeping the chunk's
    first (label) line. Rare: very long abstracts or unusual text.
    """
    out = []
    for text, meta in chunks:
        if ch.tokens(text) <= MAX_CHUNK_TOKENS or "\n" not in text:
            out.append((text, meta))
            continue
        header, body = text.split("\n", 1)
        for piece in ch.splitter.split_text(body):
            out.append((f"{header}\n{piece}", meta))
    return out


def item_chunks(item, lib, ch: Chunker, stats: Counter, keep_references=False):
    """
    All (text, metadata) chunks for one Zotero item; metadata has every META_KEYS
    key except fingerprint, which is added once the item's chunks are known.
    """
    f = item["fields"]
    cite, title = item["citation"], item["title"]
    publication = (f.get("publicationTitle") or f.get("proceedingsTitle") or f.get("bookTitle")
                   or f.get("websiteTitle") or f.get("blogTitle") or f.get("publisher")
                   or f.get("university") or f.get("institution") or "")
    base = {"item_key": item["key"], "attachment_key": "", "title": title, "citation": cite,
            "authors": authors_text(item), "year": item["year"], "item_type": item["item_type"],
            "publication": publication, "collections": "; ".join(item["collections"]),
            "tags": "; ".join(item["tags"]), "doi": f.get("DOI", ""), "url": f.get("url", ""),
            "page": "", "page_index": "0", "library": lib["name"],
            "group_id": str(lib["group_id"] or "")}
    out = []

    # Reference card (skipped for standalone notes/attachments, which have no bibliographic data)
    if item["item_type"] not in ("note", "attachment"):
        # "journalArticle" -> "journal article"
        type_name = re.sub(r"(?<!^)([A-Z])", r" \1", item["item_type"]).lower()
        lines = [f"[{cite}] {title}", f"Type: {type_name}"]
        for label, value in (("Authors", authors_text(item, MAX_CARD_AUTHORS)), ("Year", item["year"]),
                             ("Published in", publication), ("DOI", base["doi"]),
                             ("Collections", base["collections"]), ("Tags", base["tags"])):
            if value:
                lines.append(f"{label}: {value}")
        card = "\n".join(lines)
        abstract = (f.get("abstractNote") or "").strip()
        if abstract:
            pieces = ch.split(abstract)
            out.append((f"{card}\nAbstract: {pieces[0]}", {**base, "chunk_type": "reference"}))
            for piece in pieces[1:]:
                out.append((f"[{cite}] {title} (abstract, continued)\n{piece}",
                            {**base, "chunk_type": "reference"}))
        else:
            out.append((card, {**base, "chunk_type": "reference"}))

    # Attachment full text
    for att in item["attachments"]:
        res = zr.attachment_text(att, lib["slug"])
        stats[f"attachments {res['status']}"] += 1
        pages = res["pages"] if keep_references else drop_references(res["pages"])
        for page, index, text in fulltext_pieces(pages, ch):
            where = f", p. {page}" if page else ""
            out.append((f"[{cite}{where}] {title}\n{text}",
                        {**base, "chunk_type": "fulltext", "attachment_key": att["key"],
                         "page": str(page), "page_index": str(index)}))

    # Notes
    for note in item["notes"]:
        if len(note["text"].split()) < 5:
            continue
        label = "your note" if item["item_type"] != "note" else "note"
        for piece in ch.split(note["text"]):
            out.append((f"[{cite}] {title} ({label})\n{piece}", {**base, "chunk_type": "note"}))

    # Highlights and comments, grouped per attachment in reading order
    by_att = {}
    for a in item["annotations"]:
        if a["text"] or a["comment"]:
            by_att.setdefault(a["attachment_key"], []).append(a)
    for att_key, annots in by_att.items():
        lines = []
        for a in annots:
            line = f"p. {a['page']}: " if a["page"] else ""
            line += f"“{a['text']}”" if a["text"] else ""
            if a["comment"]:
                line += f" — Comment: {a['comment']}" if a["text"] else f"Comment: {a['comment']}"
            lines.append(line)
        for piece in ch.split("\n".join(lines)):
            first_page = re.match(r"p\. (\S+):", piece)
            out.append((f"[{cite}] {title} (your highlights and comments)\n{piece}",
                        {**base, "chunk_type": "annotation", "attachment_key": att_key,
                         "page": first_page.group(1) if first_page else ""}))
    return enforce_max_tokens(out, ch)


def fingerprint(chunks) -> str:
    """Short hash of an item's chunks (text and metadata).

    Stored on every chunk; when it changes, the item's chunks are re-embedded.
    """
    h = hashlib.sha1()
    for text, meta in chunks:
        h.update(text.encode())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ indexing

def build(lib, source, args):
    """Bring indexes/<slug>/ up to date with one library."""
    out = INDEX_ROOT / lib["slug"]
    info_path = out / "BUILD_INFO.json"
    if out.exists() and info_path.exists() and not args.rebuild:
        info = json.loads(info_path.read_text())
        if (info.get("embed_model"), info.get("doc_prefix")) != (EMBEDDING_MODEL, EMBED_DOC_PREFIX):
            raise SystemExit(f"{out} was built with {info.get('embed_model')!r} / {info.get('doc_prefix')!r}; "
                             f"the .env now says {EMBEDDING_MODEL!r} / {EMBED_DOC_PREFIX!r}. "
                             f"Use --rebuild to re-embed it.")
        # Nothing changed in Zotero since the last build (the local API's library version
        # is the same) and no attachment files were waiting to sync: nothing to do.
        # Files arriving from sync don't change the version, hence the "missing" check.
        version = source.library_version(lib)
        if (version is not None and version == info.get("library_version") and not args.dry_run
                and not info.get("attachments", {}).get("missing")
                and info.get("target_tokens") == TARGET_TOKENS
                and info.get("keep_references") == args.keep_references):
            log.info(f"{lib['name']}: unchanged since {info['built']} (library version {version}); skipped")
            return

    t0 = time.time()
    ch = Chunker()
    stats = Counter()
    items = source.read_library(lib)
    log.info(f"{lib['name']}: {len(items)} items; reading attachments (cached text is reused)…")
    wanted = {}
    for i, item in enumerate(items, 1):
        chunks = item_chunks(item, lib, ch, stats, args.keep_references)
        fp = fingerprint(chunks)
        wanted[item["key"]] = [(t, {**m, "fingerprint": fp}) for t, m in chunks]
        if i % 250 == 0:
            log.info(f"  {i}/{len(items)} items read")
    all_chunks = [c for cs in wanted.values() for c in cs]
    kinds = Counter(m["chunk_type"] for _, m in all_chunks)
    sizes = sorted(ch.tokens(t) for t, _ in all_chunks) or [0]
    log.info(f"  {len(all_chunks)} chunks {dict(kinds)}; tokens median {sizes[len(sizes) // 2]}, "
             f"max {sizes[-1]}")
    log.info(f"  attachments: {dict(sorted((k[12:], v) for k, v in stats.items()))} "
             f"(missing = file not synced/downloaded yet)")
    if args.dry_run:
        return

    import lancedb
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.lancedb import LanceDBVectorStore

    if args.rebuild and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out))

    # What's already indexed: item_key -> {fingerprints}, and the row ids per item
    have_fp, have_ids = {}, {}
    if _has_table(db):
        df = db.open_table(TABLE).to_arrow().select(["id", "metadata"]).to_pandas()
        for rid, meta in zip(df["id"], df["metadata"]):
            k = meta.get("item_key", "")
            have_fp.setdefault(k, set()).add(meta.get("fingerprint", ""))
            have_ids.setdefault(k, []).append(rid)

    todo = [k for k, cs in wanted.items() if cs and have_fp.get(k) != {cs[0][1]["fingerprint"]}]
    stale = [k for k in have_ids if k not in wanted or k in todo or not wanted[k]]
    log.info(f"  {len(todo)} items to (re-)embed, {len([k for k in stale if k not in todo])} to remove, "
             f"{len(wanted) - len(todo)} unchanged")

    embed = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL, keep_alive=-1,
                            text_instruction=EMBED_DOC_PREFIX or None, embed_batch_size=32)
    store = None
    done_chunks = 0
    total = sum(len(wanted[k]) for k in todo)
    for b in range(0, len(todo), EMBED_BATCH_ITEMS):
        keys = todo[b:b + EMBED_BATCH_ITEMS]
        batch = [(k, n, t, m) for k in keys for n, (t, m) in enumerate(wanted[k])]
        vectors = embed.get_text_embedding_batch([t for _, _, t, _ in batch])
        nodes = []
        for (k, n, text, meta), vec in zip(batch, vectors):
            node = TextNode(text=text, metadata=meta, id_=f"{k}:{meta['fingerprint']}:{n}", embedding=vec)
            node.excluded_embed_metadata_keys = list(meta.keys())   # embed the chunk text only
            nodes.append(node)
        # Drop these items' old chunks first: if the run stops before the add below,
        # the items are simply missing and the next run adds them again
        _delete_ids(db, [rid for k in keys for rid in have_ids.get(k, [])])
        if store is None:
            mode = "append" if _has_table(db) else "overwrite"
            store = LanceDBVectorStore(uri=str(out), table_name=TABLE, mode=mode)
        store.add(nodes)
        done_chunks += len(batch)
        log.info(f"  embedded {done_chunks}/{total} chunks ({b + len(keys)}/{len(todo)} items, "
                 f"{time.time() - t0:.0f}s)")
    _delete_ids(db, [rid for k in stale if k not in todo for rid in have_ids[k]])

    if _has_table(db) and (todo or stale):
        _compact(db)
    rows = db.open_table(TABLE).count_rows() if _has_table(db) else 0
    info_path.write_text(json.dumps({
        "library": lib["name"], "slug": lib["slug"], "library_id": lib["library_id"],
        "group_id": lib["group_id"], "items": len(items), "chunks": rows,
        "source": source.name, "library_version": source.library_version(lib),
        "chunk_types": dict(kinds), "attachments": {k[12:]: v for k, v in stats.items()},
        "built": datetime.now().isoformat(timespec="seconds"), "embed_model": EMBEDDING_MODEL,
        "doc_prefix": EMBED_DOC_PREFIX, "target_tokens": TARGET_TOKENS,
        "keep_references": args.keep_references}, indent=1))
    log.info(f"✓ {lib['name']}: {rows} chunks in {out} ({time.time() - t0:.0f}s)")


def _compact(db):
    """
    Merge the table's data files and delete superseded versions.

    LanceDB never overwrites in place: every add or delete writes new files and
    keeps the old versions, so without this an index grows with each update.
    """
    from datetime import timedelta
    try:
        db.open_table(TABLE).optimize(cleanup_older_than=timedelta(seconds=0))
    except Exception as e:   # only costs disk space; never fail the build over it
        log.warning(f"  could not compact the index: {e}")


def _has_table(db) -> bool:
    """True if the index folder already contains the chunks table."""
    try:
        db.open_table(TABLE)
        return True
    except Exception:
        return False


def _delete_ids(db, ids):
    """Delete rows by id, in batches so the SQL filter stays a reasonable size."""
    if not ids:
        return
    tbl = db.open_table(TABLE)
    for i in range(0, len(ids), 500):
        quoted = ", ".join("'" + str(x).replace("'", "''") + "'" for x in ids[i:i + 500])
        tbl.delete(f"id IN ({quoted})")


def main():
    """Command-line entry point (see the module docstring for usage)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list libraries and exit")
    ap.add_argument("--library", action="append", default=[], help="library name, group ID or slug")
    ap.add_argument("--all", action="store_true", help="every library that has items")
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the index from scratch")
    ap.add_argument("--dry-run", action="store_true", help="read and chunk only; no embedding")
    ap.add_argument("--keep-references", action="store_true", help="also index papers' reference lists")
    args = ap.parse_args()

    source = zr.open_source()
    libs = source.list_libraries()
    if args.list or not (args.library or args.all):
        built = {p.parent.name for p in INDEX_ROOT.glob("*/BUILD_INFO.json")}
        print(f"{'slug':20} {'items':>6}  {'indexed':7}  name")
        for lib in libs:
            print(f"{lib['slug']:20} {lib['items']:6}  {'yes' if lib['slug'] in built else '':7}  {lib['name']}")
        if not args.list:
            print("\nChoose with --library NAME|GROUP_ID|SLUG (repeatable) or --all")
        return

    chosen = [lib for lib in libs if lib["items"]] if args.all else []
    for w in args.library:
        lib = zr.find_library(libs, w)
        if not lib:
            raise SystemExit(f"No library matches {w!r}; see --list")
        chosen.append(lib)
    for lib in chosen:
        build(lib, source, args)


if __name__ == "__main__":
    main()
