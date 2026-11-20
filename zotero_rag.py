#!/usr/bin/env python3
"""
Zotero RAG Engine
-----------------
Answers questions about one Zotero library index (built by build_index.py),
using a hybrid retrieval pipeline that runs entirely on this computer:

  follow-up question -> rewritten as a standalone question (when there is history)
    -> hybrid search: EmbeddingGemma vectors + BM25, reciprocal-rank fusion,
       with LLM-written query variations (optionally within one collection)
    -> cross-encoder reranker keeps the best few above a relevance cutoff
       (at most MAX_CHUNKS_PER_ITEM per item, so one long paper can't crowd out the rest)
    -> local LLM (Ollama) streams an answer citing [Author Year, p. N]

The models are loaded once per process and shared by every library; each
library's index (vectors + BM25) is loaded on first use (LibraryIndex).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

import json  # noqa: E402

import lancedb  # noqa: E402
from llama_index.core import Settings  # noqa: E402
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever  # noqa: E402
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode  # noqa: E402
from llama_index.embeddings.ollama import OllamaEmbedding  # noqa: E402
from llama_index.llms.ollama import Ollama  # noqa: E402
from llama_index.retrievers.bm25 import BM25Retriever  # noqa: E402

from zotero_reader import zotero_links  # noqa: E402

log = logging.getLogger("ZoteroRAG")


def _env_bool(name: str, default: str) -> bool:
    """Read a true/false setting from the environment ("1", "true" or "yes" count as true)."""
    return os.getenv(name, default).lower() in ("1", "true", "yes")


INDEX_ROOT = APP_DIR / os.getenv("INDEX_ROOT", "indexes")
TABLE = "zotero_chunks"
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b-it-q4_K_M")
LLM_THINKING = _env_bool("LLM_THINKING", "false")
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "16384"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "task: search result | query:")
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "title: none | text:")
OLLAMA_BASE_URL = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "20"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "25"))
NUM_QUERY_VARIATIONS = int(os.getenv("NUM_QUERY_VARIATIONS", "3"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-base-v1")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.1"))
MAX_CHUNKS_PER_ITEM = int(os.getenv("MAX_CHUNKS_PER_ITEM", "3"))
HISTORY_TURNS = 3   # earlier exchanges used to understand a follow-up question

ANSWER_PROMPT = """You are a research assistant helping a scholar explore the papers, notes and highlights in their Zotero library "{library}"{scope}.

Answer the question using ONLY the excerpts below. Each excerpt begins with a citation label in square brackets, such as [Smith et al. 2019, p. 4].

Rules:
1. Support every claim with the label of the excerpt it comes from, copied exactly, e.g. [Smith et al. 2019, p. 4]. Cite several excerpts when they agree, and point out where they disagree.
2. Use only the excerpts. If they do not answer the question, say so and say what is missing. Never use outside knowledge and never invent sources or labels.
3. Excerpts marked "(your note)" or "(your highlights and comments)" were written by the user: call them "your notes" or "your highlights".
4. Give the direct answer first, then the supporting detail. Use bullet points for lists of works or findings. Be concise.
5. If the question asks what the library has on a topic, list the relevant works with their labels and one line on each.

Excerpts:
---------------------
{context}
---------------------
{history}
Question: {question}

Answer:"""

CONDENSE_PROMPT = """Rewrite the user's follow-up question as a single standalone question that can be understood without the conversation (resolve words like "it", "they", "that paper", "those authors"). If it is already standalone, repeat it unchanged. Output only the question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


# ------------------------------------------------------------------ shared models

_models = {}


def init_models():
    """Configure the LLM and embedding model (once per process) and load the reranker."""
    if _models:
        return _models
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300.0,
                          keep_alive=-1, context_window=LLM_CONTEXT_WINDOW, thinking=LLM_THINKING)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL,
                                           keep_alive=-1, query_instruction=EMBED_QUERY_PREFIX or None,
                                           text_instruction=EMBED_DOC_PREFIX or None)
    _models["reranker"] = CrossEncoderReranker(RERANKER_MODEL) if RERANKER_MODEL else None
    _models["ready"] = True
    log.info(f"Models: LLM {LLM_MODEL}, embeddings {EMBEDDING_MODEL}, reranker {RERANKER_MODEL or 'none'}")
    return _models


class CrossEncoderReranker:
    """
    Scores (question, chunk) pairs 0-1 with a sentence-transformers cross-encoder.

    A cross-encoder reads the question and a passage together, so it judges
    relevance far better than the embedding search that found the candidates,
    but it is too slow to run over a whole library: it only re-scores the
    ~25 candidates from hybrid search. Ollama can't run rerankers, so this
    uses PyTorch directly (Apple GPU, NVIDIA GPU, or CPU).
    """

    def __init__(self, model_name: str):
        """Load the cross-encoder on a GPU if there is one (NVIDIA "cuda" or Apple "mps"), else the CPU.

        Args:
            model_name: Hugging Face model id, e.g. "mixedbread-ai/mxbai-rerank-base-v1".
        """
        import torch
        from sentence_transformers import CrossEncoder
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        self._sigmoid = torch.nn.Sigmoid()
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, nodes: List[NodeWithScore]) -> List[float]:
        """Relevance of each chunk to the query.

        Returns:
            One score per node, in the same order, from 0 (unrelated) to 1.
        """
        if not nodes:
            return []
        pairs = [(query, n.node.get_content(metadata_mode=MetadataMode.NONE)) for n in nodes]
        return [float(s) for s in self.model.predict(pairs, batch_size=16, show_progress_bar=False,
                                                     activation_fn=self._sigmoid)]


# "What does my library say about X?", "Which papers discuss X", "Find articles on X"...
# Framing like this drags every reranker score to ~0 (measured: 0.35 -> 0.01 for the
# same relevant abstract), so search and reranking use just the topic, X.
_TOPIC_AFTER = re.compile(r"^(.{0,80}?)\b(about|on|regarding|concerning|discuss(?:es|ing)?|cover(?:s|ing)?|"
                          r"address(?:es|ing)?|related to)\s+(.+)$", re.I)
_LIBRARY_WORDS = re.compile(r"\b(library|collection|zotero|papers?|sources?|articles?|readings?|literature|"
                            r"studies|works|items|references|i have|my notes|my highlights)\b", re.I)


def focus_query(question: str) -> str:
    """The topic of a question, without "what does my library say about"-style framing."""
    m = _TOPIC_AFTER.match(question.strip())
    if m and _LIBRARY_WORDS.search(m.group(1)) and len(m.group(3).split()) >= 2:
        return m.group(3).rstrip(" ?.")
    return question


def in_collection(collections: str, choice: str) -> bool:
    """True if a chunk's "; "-joined collection paths include `choice` or one of its subcollections."""
    return any(c == choice or c.startswith(choice + " / ") for c in collections.split("; ") if c)


def _clean_meta(meta) -> Dict:
    """Chunk metadata without LlamaIndex's internal keys (those starting with '_') or empty values."""
    return {k: v for k, v in meta.items() if not k.startswith("_") and v is not None} \
        if isinstance(meta, dict) else {}


class LanceVectorRetriever(BaseRetriever):
    """Vector search on the LanceDB table, optionally limited to one collection."""

    def __init__(self, table, top_k: int, collection: Optional[str] = None):
        """Args:
            table: The LanceDB table of chunks.
            top_k: Number of results to return.
            collection: Collection path to search within (None = whole library).
        """
        super().__init__()
        self._table, self._top_k, self._collection = table, top_k, collection

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Embed the query and return the top_k nearest chunks.

        With a collection, LanceDB pre-filters on a substring match, then exact
        membership (the collection or a subcollection) is checked here.
        """
        vec = Settings.embed_model.get_query_embedding(query_bundle.query_str)
        search = self._table.search(vec).select(["id", "text", "metadata", "_distance"])
        if self._collection:
            like = self._collection.replace("'", "''")
            search = search.where(f"metadata.collections LIKE '%{like}%'", prefilter=True)
        out = []
        for r in search.limit(self._top_k * (2 if self._collection else 1)).to_list():
            meta = _clean_meta(r["metadata"])
            if self._collection and not in_collection(meta.get("collections", ""), self._collection):
                continue
            out.append(NodeWithScore(node=TextNode(id_=r["id"], text=r["text"], metadata=meta),
                                     score=1.0 - float(r["_distance"])))
        return out[:self._top_k]


# ------------------------------------------------------------------ one library

@dataclass
class Source:
    """One excerpt the answer was based on, for the "Sources" panel."""
    label: str
    title: str
    citation: str
    chunk_type: str
    page: str
    score: float
    text: str
    links: Dict[str, str]
    url: str = ""
    doi: str = ""
    collections: str = ""


@dataclass
class Answer:
    """A streamed answer: iterate .stream for text; .sources are the excerpts it used."""
    question: str
    search_query: str
    sources: List[Source]
    stream: Iterator[str]
    variations: List[str] = field(default_factory=list)


def list_indexes() -> List[Dict]:
    """BUILD_INFO of every built index, My Library first then by name."""
    out = []
    for p in INDEX_ROOT.glob("*/BUILD_INFO.json"):
        info = json.loads(p.read_text())
        info["path"] = str(p.parent)
        if info.get("chunks"):
            out.append(info)
    return sorted(out, key=lambda i: (i["slug"] != "my-library", i["library"].lower()))


class LibraryIndex:
    """Retrieval and answering for one Zotero library's index."""

    def __init__(self, path: str):
        """Load the LanceDB table and build the BM25 index over all its chunks (a few seconds)."""
        self.path = Path(path)
        self.info = json.loads((self.path / "BUILD_INFO.json").read_text())
        if self.info.get("doc_prefix") != EMBED_DOC_PREFIX or self.info.get("embed_model") != EMBEDDING_MODEL:
            raise ValueError(f"Index {self.path.name} was built with {self.info.get('embed_model')} / "
                             f"{self.info.get('doc_prefix')!r}; .env says {EMBEDDING_MODEL} / {EMBED_DOC_PREFIX!r}. "
                             f"Rebuild it with build_index.py --rebuild.")
        init_models()
        self.name = self.info["library"]
        self.group_id = self.info.get("group_id")
        self.table = lancedb.connect(str(self.path)).open_table(TABLE)
        df = self.table.to_arrow().select(["id", "text", "metadata"]).to_pandas()
        self.nodes = [TextNode(id_=str(i), text=str(t), metadata=_clean_meta(m))
                      for i, t, m in zip(df["id"], df["text"], df["metadata"])]
        cols = set()
        for n in self.nodes:
            for c in n.metadata.get("collections", "").split("; "):
                parts = [p for p in c.split(" / ") if p]
                cols.update(" / ".join(parts[:k]) for k in range(1, len(parts) + 1))
        self.collections = sorted(cols, key=str.lower)
        self._retrievers = {}
        log.info(f"Loaded {self.name}: {len(self.nodes)} chunks, {len(self.collections)} collections")

    def _retriever(self, collection: Optional[str]) -> QueryFusionRetriever:
        """Hybrid retriever for the whole library or one collection (built on first use, then kept)."""
        if collection not in self._retrievers:
            nodes = self.nodes if not collection else [
                n for n in self.nodes if in_collection(n.metadata.get("collections", ""), collection)]
            if not nodes:
                raise ValueError(f"No indexed items in collection {collection!r}")
            retrievers = [LanceVectorRetriever(self.table, CANDIDATE_TOP_K, collection),
                          BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(CANDIDATE_TOP_K, len(nodes)))]
            self._retrievers[collection] = QueryFusionRetriever(
                retrievers=retrievers, similarity_top_k=TOP_K_RETRIEVAL, num_queries=NUM_QUERY_VARIATIONS,
                mode="reciprocal_rerank", use_async=False)
        return self._retrievers[collection]

    def retrieve(self, query: str, collection: Optional[str] = None) -> List[NodeWithScore]:
        """
        Hybrid search, de-duplication, reranking and the per-item cap.

        Returns:
            The excerpts to answer from, best first; empty when nothing passes
            RERANK_MIN_SCORE (the question isn't covered by this library).
        """
        seen, nodes = set(), []
        for n in self._retriever(collection).retrieve(query):
            # Compare the passage body, not the "[Author Year, p. N] Title" header, so the
            # same PDF saved under two Zotero items (it happens) only takes one slot
            text = n.node.get_content()
            key = (text.split("\n", 1)[1] if "\n" in text else text)[:300]
            if key not in seen:
                seen.add(key)
                nodes.append(n)
        reranker = _models.get("reranker")
        if reranker:
            for n, s in zip(nodes, reranker.score(query, nodes)):
                n.score = s
            nodes = [n for n in sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
                     if (n.score or 0.0) >= RERANK_MIN_SCORE]
        per_item, kept = {}, []
        for n in nodes:
            k = n.node.metadata.get("item_key", "")
            if per_item.get(k, 0) < MAX_CHUNKS_PER_ITEM:
                per_item[k] = per_item.get(k, 0) + 1
                kept.append(n)
        return kept[:RERANK_TOP_N]

    def _standalone(self, question: str, history: List[Dict]) -> str:
        """Rewrite a follow-up ("what method did they use?") so search can find it on its own."""
        if not history:
            return question
        prompt = CONDENSE_PROMPT.format(history=_format_history(history), question=question)
        try:
            text = Settings.llm.complete(prompt).text.strip().strip('"').split("\n")[0]
            return text or question
        except Exception as e:
            log.warning(f"Could not rewrite follow-up question: {e}")
            return question

    def ask(self, question: str, history: Optional[List[Dict]] = None,
            collection: Optional[str] = None) -> Answer:
        """
        Retrieve excerpts and start streaming the answer.

        Args:
            question: The user's question.
            history: Earlier turns, [{"role": "user"|"assistant", "content": str}].
            collection: Limit the search to this collection path (and its subcollections).
        """
        history = history or []
        search_query = focus_query(self._standalone(question, history))
        nodes = self.retrieve(search_query, collection)
        sources = [self._source(n) for n in nodes]
        if not nodes:
            where = f"the collection “{collection}”" if collection else f"the “{self.name}” library"
            msg = (f"I couldn't find anything in {where} that answers this. Try rephrasing, "
                   f"or choose a different library or collection.")
            return Answer(question, search_query, [], iter([msg]))
        context = "\n\n".join(n.node.get_content(metadata_mode=MetadataMode.NONE) for n in nodes)
        hist = _format_history(history)
        prompt = ANSWER_PROMPT.format(
            library=self.name, scope=f', collection "{collection}"' if collection else "",
            context=context, question=question,
            history=f"\nConversation so far (for context only):\n{hist}\n" if hist else "")

        def stream():
            """Yield the answer text as the LLM generates it."""
            for chunk in Settings.llm.stream_complete(prompt):
                if chunk.delta:
                    yield chunk.delta

        return Answer(question, search_query, sources, stream())

    def _source(self, n: NodeWithScore) -> Source:
        """Turn a retrieved chunk into a Source for display (label, excerpt, Zotero links)."""
        m = n.node.metadata
        text = n.node.get_content(metadata_mode=MetadataMode.NONE)
        label = text.split("\n", 1)[0]
        label = label[1:label.index("]")] if label.startswith("[") and "]" in label else m.get("citation", "")
        try:
            page_index = int(m.get("page_index") or 0)
        except ValueError:
            page_index = 0
        return Source(label=label, title=m.get("title", ""), citation=m.get("citation", ""),
                      chunk_type=m.get("chunk_type", ""), page=m.get("page", ""), score=n.score or 0.0,
                      text=text.split("\n", 1)[1] if "\n" in text else text,
                      links=zotero_links(self.group_id, m.get("item_key", ""), m.get("attachment_key", ""),
                                         page_index),
                      url=m.get("url", ""), doi=m.get("doi", ""), collections=m.get("collections", ""))


def _format_history(history: List[Dict]) -> str:
    """The last HISTORY_TURNS exchanges as 'User: ... / Assistant: ...' lines.

    Long answers are truncated: they only help to resolve follow-up questions.
    """
    lines = []
    for turn in history[-HISTORY_TURNS * 2:]:
        content = turn["content"]
        if turn["role"] == "assistant" and len(content) > 800:
            content = content[:800] + "…"
        lines.append(f"{'User' if turn['role'] == 'user' else 'Assistant'}: {content}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    idx = list_indexes()
    lib = LibraryIndex(next(i["path"] for i in idx if sys.argv[1] in (i["slug"], i["library"])))
    ans = lib.ask(" ".join(sys.argv[2:]))
    for s in ans.sources:
        print(f"  {s.score:.2f} [{s.label}] {s.title}")
    for t in ans.stream:
        print(t, end="", flush=True)
    print()
