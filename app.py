#!/usr/bin/env python3
"""
Zotero Chat — Streamlit front end
---------------------------------
Chat with one Zotero library at a time. The sidebar lists every index built by
build_index.py (one per library) and, for the chosen library, its collections,
so a question can be limited to one collection and its subcollections.

Each library keeps its own conversation, so switching libraries and back picks
up where you left off. Answers cite [Author Year, p. N]; the Sources panel under
each answer shows the excerpts used, with links that open the item in Zotero or
the PDF at that page.

Run:  ./start-app.sh   (or: venv/bin/streamlit run app.py --server.port 8502)
"""

import html
import logging
from datetime import datetime

import streamlit as st

import zotero_rag as zrag

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="Zotero Chat", page_icon="📚", layout="wide")
st.markdown("""<style>
  .src { border-left: 3px solid #cc2936; padding: .35rem .75rem; margin: .5rem 0 .9rem 0;
         background: rgba(127,127,127,.06); border-radius: 0 6px 6px 0; }
  .src .meta { font-size: .82rem; opacity: .75; }
  .src .excerpt { font-size: .86rem; white-space: pre-wrap; margin-top: .3rem; }
  .src a { margin-right: .9rem; font-size: .85rem; }
</style>""", unsafe_allow_html=True)

CHUNK_LABELS = {"reference": "reference & abstract", "fulltext": "full text", "note": "your note",
                "annotation": "your highlights"}


@st.cache_resource(show_spinner=False)
def load_index(path: str, built: str) -> zrag.LibraryIndex:
    """One LibraryIndex per index folder, kept across reruns ('built' reloads it after a rebuild)."""
    return zrag.LibraryIndex(path)


def render_sources(sources, search_query: str, question: str):
    """Expandable list of the excerpts an answer used, with Zotero and web links."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        if search_query.strip().lower() != question.strip().lower():
            st.caption(f"Searched for: “{search_query}”")
        for s in sources:
            links = [f'<a href="{html.escape(s.links["select"])}">Show in Zotero</a>']
            if "open" in s.links and s.chunk_type in ("fulltext", "annotation"):
                where = f" at p. {html.escape(s.page)}" if s.page else ""
                links.append(f'<a href="{html.escape(s.links["open"])}">Open PDF{where}</a>')
            if s.doi:
                links.append(f'<a href="https://doi.org/{html.escape(s.doi)}" target="_blank">DOI</a>')
            elif s.url:
                links.append(f'<a href="{html.escape(s.url)}" target="_blank">Web</a>')
            excerpt = s.text if len(s.text) <= 700 else s.text[:700] + "…"
            st.markdown(
                f'<div class="src"><b>[{html.escape(s.label)}]</b> {html.escape(s.title)}'
                f'<div class="meta">{CHUNK_LABELS.get(s.chunk_type, s.chunk_type)} · relevance {s.score:.2f}'
                f'{" · " + html.escape(s.collections) if s.collections else ""}</div>'
                f'<div class="excerpt">{html.escape(excerpt)}</div>'
                f'<div>{" ".join(links)}</div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar

indexes = zrag.list_indexes()
if not indexes:
    st.title("📚 Zotero Chat")
    st.warning("No library indexes yet. Build one in a terminal, e.g.\n\n"
               "```\nvenv/bin/python build_index.py --list\n"
               "venv/bin/python build_index.py --library \"My Library\"\n```")
    st.stop()

with st.sidebar:
    st.header("📚 Zotero Chat")
    by_slug = {i["slug"]: i for i in indexes}
    slug = st.selectbox(
        "Library", list(by_slug), key="library",
        format_func=lambda s: f"{by_slug[s]['library']}  ({by_slug[s]['items']:,} items)")
    choice = by_slug[slug]
    try:
        with st.spinner(f"Loading {choice['library']}…"):
            lib = load_index(choice["path"], choice["built"])
    except Exception as e:
        st.error(str(e))
        st.stop()

    collection = st.selectbox("Collection", ["All collections"] + lib.collections,
                              key=f"collection-{choice['slug']}",
                              help="Search only this collection and its subcollections.")
    collection = None if collection == "All collections" else collection

    att = choice.get("attachments", {})
    b = datetime.fromisoformat(choice["built"])
    built = f"{b:%b} {b.day}, {b:%Y} {b.hour % 12 or 12}:{b:%M %p}"   # portable (no %-d on Windows)
    types = choice.get("chunk_types", {})
    st.caption(f"{choice['chunks']:,} passages · {types.get('fulltext', 0):,} from full text · "
               f"{types.get('note', 0) + types.get('annotation', 0):,} from your notes and highlights  \n"
               f"Index updated {built}")
    if att.get("missing"):
        st.caption(f"⚠️ {att['missing']:,} attachment files weren't on this Mac when the index was built "
                   f"(not synced yet?). Re-run `build_index.py --library {choice['slug']}` to add them.")

    chats = st.session_state.setdefault("chats", {})
    if st.button("New conversation", use_container_width=True):
        chats[choice["slug"]] = []
        st.rerun()
    st.caption("Update indexes after changing your Zotero library:  \n"
               "`venv/bin/python build_index.py --all`")

# ------------------------------------------------------------------ chat

messages = chats.setdefault(choice["slug"], [])
scope = f" › {collection}" if collection else ""
st.subheader(f"{lib.name}{scope}")

if not messages:
    st.caption("Ask about the papers, notes and highlights in this library. Examples: "
               "*What does my library say about…?* · *Which papers use survey methods?* · "
               "*Summarize my notes on…* · *How do the sources define…?*")

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), m.get("search_query", ""), m.get("question", ""))

if question := st.chat_input(f"Ask {lib.name}{scope}…"):
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your library…"):
                ans = lib.ask(question, history=history, collection=collection)
                first = next(ans.stream, "")   # keep the spinner up until text is ready
            def rest():
                """Yield the first chunk (already fetched under the spinner), then the rest of the stream."""
                yield first
                yield from ans.stream
            text = st.write_stream(rest())
        except Exception as e:
            logging.exception("Answer failed")
            text = f"Sorry, something went wrong: {e}"
            st.error(text)
            ans = None
        sources = ans.sources if ans else []
        render_sources(sources, ans.search_query if ans else "", question)
    messages.append({"role": "assistant", "content": text, "sources": sources,
                     "search_query": ans.search_query if ans else "", "question": question})
