#!/usr/bin/env python3
"""
Zotero Library Reader
---------------------
Returns, per Zotero library, every item with its metadata, attachments' text,
notes and PDF annotations. Two interchangeable sources produce identical
records (open_source() picks one; ZOTERO_SOURCE in .env can force either):

  - ApiSource (preferred): Zotero's local API, http://localhost:23119/api/
    (Zotero 7+, "Allow other applications on this computer to communicate with
    Zotero" in Settings > Advanced). The supported interface, so it keeps working
    across Zotero updates, and its library version numbers let an index update
    skip unchanged libraries. Needs Zotero to be running.
  - SqliteSource (fallback when Zotero is closed): reads a snapshot copy of
    ~/Zotero/zotero.sqlite (Zotero locks the live file while it runs). Reads
    Zotero's internal tables, which a future Zotero version could change.

Neither ever modifies anything in Zotero. Attachment files are read straight
from ~/Zotero/storage/ (or their linked location) in both cases.

Library slugs, used for index folder names: "my-library" and
"group-<groupID>" (stable if a group is renamed).

Attachment text (PDF pages, HTML snapshots, plain text) is cached in
cache/text/<slug>/<attachmentKey>.json, keyed by the file's size and
modification time, so rebuilding an index only re-reads changed files.
"""

import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("ZoteroReader")

APP_DIR = Path(__file__).resolve().parent
ZOTERO_DATA_DIR = Path(os.path.expanduser(os.getenv("ZOTERO_DATA_DIR", "~/Zotero")))
TEXT_CACHE = APP_DIR / "cache" / "text"
# Very long PDFs (whole books, proceedings) are cut off here to keep indexing time sane
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "600"))

# itemAnnotations.type
ANNOTATION_TYPES = {1: "highlight", 2: "note", 3: "image", 4: "ink", 5: "underline", 6: "text"}
# itemAttachments.linkMode
IMPORTED_FILE, IMPORTED_URL, LINKED_FILE, LINKED_URL = 0, 1, 2, 3


# ------------------------------------------------------------------ database

def open_snapshot(data_dir: Path = ZOTERO_DATA_DIR, retries: int = 3) -> sqlite3.Connection:
    """
    Copy zotero.sqlite (+ its -wal file) to a temp folder and open the copy.

    The copy is taken while Zotero may be writing (e.g. during sync), so it is
    checked and re-taken if it comes out inconsistent.

    Returns:
        A sqlite3 connection with row access by column name.
    """
    src = data_dir / "zotero.sqlite"
    if not src.exists():
        raise FileNotFoundError(f"No Zotero database at {src}. Set ZOTERO_DATA_DIR in .env.")
    for attempt in range(1, retries + 1):
        tmp = Path(tempfile.mkdtemp(prefix="zotero-snapshot-"))
        dst = tmp / "zotero.sqlite"
        shutil.copy2(src, dst)
        wal = data_dir / "zotero.sqlite-wal"
        if wal.exists():
            shutil.copy2(wal, tmp / "zotero.sqlite-wal")
        conn = sqlite3.connect(dst)
        conn.row_factory = sqlite3.Row
        try:
            if conn.execute("PRAGMA quick_check").fetchone()[0] == "ok":
                return conn
        except sqlite3.DatabaseError:
            pass
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)
        log.warning(f"Snapshot {attempt} was inconsistent (Zotero busy writing?); retrying")
        time.sleep(3)
    raise RuntimeError("Could not take a consistent snapshot of the Zotero database")


def list_libraries(conn: sqlite3.Connection) -> List[Dict]:
    """
    Every library in the database with its number of (non-deleted) regular items.

    Returns:
        [{library_id, group_id, name, slug, items}], My Library first. items counts
        top-level items (regular items plus standalone notes/attachments). slug is the
        index folder name: "my-library" or "group-<groupID>" (stable if renamed).
    """
    rows = conn.execute(f"""
        SELECT l.libraryID, l.type, g.groupID, g.name,
               (SELECT COUNT(*) FROM items i JOIN itemTypes t USING(itemTypeID)
                 WHERE i.libraryID = l.libraryID
                   AND t.typeName != 'annotation'
                   AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                   AND i.itemID NOT IN (SELECT itemID FROM itemAttachments WHERE parentItemID IS NOT NULL)
                   AND i.itemID NOT IN (SELECT itemID FROM itemNotes WHERE parentItemID IS NOT NULL)) AS n
        FROM libraries l LEFT JOIN groups g USING(libraryID)
        WHERE l.type IN ('user', 'group')
        ORDER BY l.type = 'group', g.name COLLATE NOCASE""").fetchall()
    out = []
    for r in rows:
        if r["type"] == "user":
            out.append({"library_id": r["libraryID"], "group_id": None, "name": "My Library",
                        "slug": "my-library", "items": r["n"]})
        else:
            out.append({"library_id": r["libraryID"], "group_id": r["groupID"], "name": r["name"],
                        "slug": f"group-{r['groupID']}", "items": r["n"]})
    return out


def _base_attachment_dir() -> Optional[Path]:
    """Zotero's "Linked Attachment Base Directory" (for "attachments:" paths), from .env or prefs.js."""
    if os.getenv("ZOTERO_BASE_DIR"):
        return Path(os.path.expanduser(os.environ["ZOTERO_BASE_DIR"]))
    home = Path.home()
    profile_dirs = [home / "Library/Application Support/Zotero/Profiles",   # macOS
                    home / ".zotero/zotero",                                 # Linux
                    Path(os.getenv("APPDATA", home)) / "Zotero/Zotero/Profiles"]  # Windows
    for prefs in (p for d in profile_dirs for p in d.glob("*/prefs.js")):
        m = re.search(r'user_pref\("extensions\.zotero\.baseAttachmentPath",\s*"(.+?)"\);',
                      prefs.read_text(errors="ignore"))
        if m:
            return Path(m.group(1).encode().decode("unicode_escape"))
    return None


def _year(date: str) -> str:
    """The four-digit year in a Zotero date string ('' if there is none)."""
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", date or "")
    return m.group(1) if m else ""


def _citation(creators: List[Dict], year: str, title: str) -> str:
    """'Smith 2019', 'Smith & Jones 2019', 'Smith et al. 2019' (authors, else editors, else title)."""
    people = [c for c in creators if c["type"] == "author"] or creators
    names = [c["last"] or c["first"] for c in people if (c["last"] or c["first"])]
    if not names:
        who = (title[:40] + "…") if len(title) > 40 else (title or "Untitled")
    elif len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} & {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who} {year or 'n.d.'}"


def html_to_text(html: str) -> str:
    """Note / snapshot HTML to plain text, keeping paragraph breaks."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    text = soup.get_text("\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def read_library(conn: sqlite3.Connection, library_id: int) -> List[Dict]:
    """
    All non-deleted items of one library, with everything needed for indexing.

    Returns:
        [{key, item_id, item_type, title, fields{}, creators[{first,last,type}],
          year, citation, tags[], collections[], attachments[], notes[], annotations[],
          date_modified}]. Standalone notes and standalone attachments (no parent
          item) are returned as items of their own. attachments carry
          {key, title, link_mode, content_type, path (resolved Path or None)}; notes
          {key, text}; annotations {attachment_key, type, text, comment, page, sort}.
    """
    q = lambda sql: conn.execute(sql, (library_id,)).fetchall()
    deleted = {r[0] for r in conn.execute("SELECT itemID FROM deletedItems")}

    items = {}
    for r in q("""SELECT i.itemID, i.key, t.typeName, i.dateModified FROM items i
                  JOIN itemTypes t USING(itemTypeID) WHERE i.libraryID = ?"""):
        if r["itemID"] not in deleted:
            items[r["itemID"]] = {"item_id": r["itemID"], "key": r["key"], "item_type": r["typeName"],
                                  "date_modified": r["dateModified"], "fields": {}, "creators": [],
                                  "tags": [], "collections": [], "attachments": [], "notes": [],
                                  "annotations": []}

    for r in q("""SELECT d.itemID, f.fieldName, v.value FROM itemData d
                  JOIN fields f USING(fieldID) JOIN itemDataValues v USING(valueID)
                  JOIN items i USING(itemID) WHERE i.libraryID = ?"""):
        if r["itemID"] in items:
            items[r["itemID"]]["fields"][r["fieldName"]] = str(r["value"])

    for r in q("""SELECT ic.itemID, c.firstName, c.lastName, ct.creatorType FROM itemCreators ic
                  JOIN creators c USING(creatorID) JOIN creatorTypes ct USING(creatorTypeID)
                  JOIN items i USING(itemID) WHERE i.libraryID = ? ORDER BY ic.itemID, ic.orderIndex"""):
        if r["itemID"] in items:
            items[r["itemID"]]["creators"].append(
                {"first": r["firstName"] or "", "last": r["lastName"] or "", "type": r["creatorType"]})

    for r in q("""SELECT it.itemID, t.name FROM itemTags it JOIN tags t USING(tagID)
                  JOIN items i USING(itemID) WHERE i.libraryID = ?"""):
        if r["itemID"] in items:
            items[r["itemID"]]["tags"].append(r["name"])

    # Collection paths ("Parent / Child"), skipping collections in the trash
    deleted_cols = {r[0] for r in conn.execute("SELECT collectionID FROM deletedCollections")}
    cols = {r["collectionID"]: r for r in q("SELECT * FROM collections WHERE libraryID = ?")}

    def col_path(cid, depth=0):
        """Full path of a collection, e.g. 'Parent / Child' (guards against cycles)."""
        c = cols[cid]
        parent = c["parentCollectionID"]
        if parent in cols and depth < 20:
            return f"{col_path(parent, depth + 1)} / {c['collectionName']}"
        return c["collectionName"]

    for r in q("""SELECT ci.collectionID, ci.itemID FROM collectionItems ci
                  JOIN collections c USING(collectionID) WHERE c.libraryID = ?"""):
        if r["itemID"] in items and r["collectionID"] not in deleted_cols:
            items[r["itemID"]]["collections"].append(col_path(r["collectionID"]))

    storage = ZOTERO_DATA_DIR / "storage"
    base_dir = None
    att_parent = {}   # attachment itemID -> parent itemID (for annotations)
    for r in q("""SELECT a.* FROM itemAttachments a JOIN items i USING(itemID) WHERE i.libraryID = ?"""):
        att = items.get(r["itemID"])
        if not att:
            continue
        path, raw = None, r["path"] or ""
        if r["linkMode"] in (IMPORTED_FILE, IMPORTED_URL) and raw.startswith("storage:"):
            path = storage / att["key"] / raw[len("storage:"):]
        elif r["linkMode"] == LINKED_FILE and raw:
            if raw.startswith("attachments:"):
                base_dir = base_dir or _base_attachment_dir()
                path = base_dir / raw[len("attachments:"):] if base_dir else None
            else:
                path = Path(raw)
        info = {"key": att["key"], "item_id": r["itemID"], "title": att["fields"].get("title", ""),
                "link_mode": r["linkMode"], "content_type": r["contentType"] or "", "path": path}
        parent = r["parentItemID"]
        if parent in items:
            items[parent]["attachments"].append(info)
            att_parent[r["itemID"]] = parent
            att["_child"] = True
        elif parent:
            att["_child"] = True                # its parent is in the trash: leave it out
        else:
            att["attachments"].append(info)     # standalone attachment: its own item
            att_parent[r["itemID"]] = r["itemID"]

    for r in q("""SELECT n.* FROM itemNotes n JOIN items i USING(itemID) WHERE i.libraryID = ?"""):
        note = items.get(r["itemID"])
        if not note:
            continue
        entry = {"key": note["key"], "text": html_to_text(r["note"])}
        if note["item_type"] == "attachment":
            # An attachment's own note: it belongs with the attachment's item
            owner = att_parent.get(r["itemID"])
            if owner in items and entry["text"]:
                items[owner]["notes"].append(entry)
            continue
        if r["parentItemID"] in items:
            items[r["parentItemID"]]["notes"].append(entry)
            note["_child"] = True
        elif r["parentItemID"]:
            note["_child"] = True               # its parent is in the trash: leave it out
        else:
            # Title = first line, as ApiSource does (the API has no note titles)
            first = entry["text"].split("\n", 1)[0].strip()
            note["fields"].setdefault("title", first[:120] or "Untitled note")
            note["notes"].append(entry)          # standalone note: its own item

    for r in q("""SELECT a.*, i.key FROM itemAnnotations a JOIN items i USING(itemID)
                  WHERE i.libraryID = ? ORDER BY a.sortIndex"""):
        owner = att_parent.get(r["parentItemID"])
        if owner in items and r["itemID"] not in deleted:
            att_key = items[r["parentItemID"]]["key"] if r["parentItemID"] in items else ""
            items[owner]["annotations"].append({
                "attachment_key": att_key, "type": ANNOTATION_TYPES.get(r["type"], "other"),
                "text": (r["text"] or "").strip(), "comment": (r["comment"] or "").strip(),
                "page": r["pageLabel"] or "", "sort": r["sortIndex"]})
        if r["itemID"] in items:
            items[r["itemID"]]["_child"] = True

    return _finish(items.values())


def _finish(items) -> List[Dict]:
    """Top-level items only, with title, year, citation label and sorted collections/tags."""
    out = []
    for it in items:
        if it.get("_child") or it["item_type"] == "annotation":
            continue
        f = it["fields"]
        it["title"] = (f.get("title") or "").strip() or "Untitled"
        it["year"] = _year(f.get("date", ""))
        it["citation"] = _citation(it["creators"], it["year"], it["title"])
        it["collections"] = sorted(set(it["collections"]))
        it["tags"] = sorted(set(it["tags"]))
        # A fixed order, so both sources yield identical chunks (and fingerprints)
        it["attachments"].sort(key=lambda a: a["key"])
        it["notes"].sort(key=lambda n: n["key"])
        it["annotations"].sort(key=lambda a: (a["attachment_key"], a["sort"]))
        it.pop("_child", None)
        out.append(it)
    return sorted(out, key=lambda i: i["key"])


# ------------------------------------------------------------------ sources

class SqliteSource:
    """Libraries and items from a snapshot of zotero.sqlite (works with Zotero closed)."""

    name = "Zotero database snapshot"

    def __init__(self):
        """Take a snapshot of zotero.sqlite and open it."""
        self.conn = open_snapshot()

    def list_libraries(self) -> List[Dict]:
        """See the module-level list_libraries()."""
        return list_libraries(self.conn)

    def read_library(self, lib: Dict) -> List[Dict]:
        """See the module-level read_library()."""
        return read_library(self.conn, lib["library_id"])

    def library_version(self, lib: Dict) -> Optional[int]:
        """Always None: the snapshot has no cheap change check, so every item is compared."""
        return None   # no cheap change check; every item is compared instead


class ApiSource:
    """
    Libraries and items from Zotero's local API (Zotero must be running).

    Same records as SqliteSource. The API leaves out items in the trash; it has
    no annotation text search or full text, so attachment text is still read
    from the files.
    """

    name = "Zotero local API"
    PAGE = 100   # the API's maximum page size

    def __init__(self, base: str = ""):
        """Connect to the local API and check that it answers.

        Args:
            base: API root URL; defaults to ZOTERO_API_URL or http://localhost:23119/api.

        Raises:
            URLError / HTTPError if Zotero isn't running or the local API is disabled.
        """
        self.base = (base or os.getenv("ZOTERO_API_URL", "http://localhost:23119/api")).rstrip("/")
        self._versions = {}
        self._get("/users/0/items/top", {"limit": 1})   # raises if Zotero / the API isn't available

    def _get(self, path: str, params: Optional[Dict] = None):
        """(JSON body, headers) for one API request."""
        import urllib.parse
        import urllib.request
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers={"Zotero-API-Version": "3"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), r.headers

    def _all(self, path: str, params: Optional[Dict] = None) -> List[Dict]:
        """Every result of a paged list request."""
        out, start = [], 0
        while True:
            body, headers = self._get(path, {**(params or {}), "limit": self.PAGE, "start": start})
            out.extend(body)
            total = int(headers.get("Total-Results", len(out)))
            start += self.PAGE
            if start >= total or not body:
                return out

    @staticmethod
    def _prefix(lib: Dict) -> str:
        """API path prefix for a library: '/users/0' (My Library) or '/groups/<id>'."""
        return f"/groups/{lib['group_id']}" if lib["group_id"] else "/users/0"

    def list_libraries(self) -> List[Dict]:
        """Same shape as list_libraries(): My Library first, then groups by name."""
        libs = [{"library_id": None, "group_id": None, "name": "My Library", "slug": "my-library"}]
        for g in self._all("/users/0/groups"):
            libs.append({"library_id": None, "group_id": g["id"], "name": g["data"]["name"],
                         "slug": f"group-{g['id']}"})
        for lib in libs:
            # Top-level items (regular items plus standalone notes/attachments), not in the trash
            _, headers = self._get(self._prefix(lib) + "/items/top", {"limit": 1})
            lib["items"] = int(headers.get("Total-Results", 0))
            self._versions[lib["slug"]] = int(headers.get("Last-Modified-Version", 0) or 0)
        return [libs[0]] + sorted(libs[1:], key=lambda l: l["name"].lower())

    def library_version(self, lib: Dict) -> Optional[int]:
        """Zotero's version number for the library; it changes whenever anything in it changes."""
        if lib["slug"] not in self._versions:
            _, headers = self._get(self._prefix(lib) + "/items/top", {"limit": 1})
            self._versions[lib["slug"]] = int(headers.get("Last-Modified-Version", 0) or 0)
        return self._versions[lib["slug"]]

    def read_library(self, lib: Dict) -> List[Dict]:
        """All non-trashed items of one library, in the same record format as read_library()."""
        prefix = self._prefix(lib)
        cols = {c["key"]: c["data"] for c in self._all(prefix + "/collections")
                if not c["data"].get("deleted")}

        def col_path(key, depth=0):
            """Full path of a collection, e.g. 'Parent / Child' (guards against cycles)."""
            c = cols[key]
            parent = c.get("parentCollection")
            if parent in cols and depth < 20:
                return f"{col_path(parent, depth + 1)} / {c['name']}"
            return c["name"]

        raw = []
        for r in self._all(prefix + "/items"):
            d = r["data"]
            if d.get("deleted"):
                continue
            # The API gives dates as typed ("1/12/23") plus meta.parsedDate ("2023-01-12");
            # the database stores both together as "2023-01-12 1/12/23". Match that.
            parsed = (r.get("meta") or {}).get("parsedDate", "")
            if parsed and d.get("date"):
                d = {**d, "date": f"{parsed} {d['date']}"}
            raw.append(d)
        items = {}
        for d in raw:
            fields = {k: str(v) for k, v in d.items() if isinstance(v, (str, int)) and not isinstance(v, bool)
                      and k not in ("key", "version", "itemType", "parentItem", "note", "linkMode",
                                    "contentType", "charset", "filename", "path", "mtime", "md5")
                      and not k.startswith("annotation")}
            creators = [{"first": c.get("firstName", ""), "last": c.get("lastName") or c.get("name", ""),
                         "type": c.get("creatorType", "author")} for c in d.get("creators", [])]
            items[d["key"]] = {
                "item_id": None, "key": d["key"], "item_type": d["itemType"],
                "date_modified": d.get("dateModified", ""), "fields": fields, "creators": creators,
                "tags": [t["tag"] for t in d.get("tags", [])],
                "collections": [col_path(k) for k in d.get("collections", []) if k in cols],
                "attachments": [], "notes": [], "annotations": [], "_data": d}

        storage = ZOTERO_DATA_DIR / "storage"
        base_dir = None
        modes = {"imported_file": IMPORTED_FILE, "imported_url": IMPORTED_URL,
                 "linked_file": LINKED_FILE, "linked_url": LINKED_URL}
        dropped = []
        for key, it in items.items():
            d = it["_data"]
            parent = d.get("parentItem")
            kind = it["item_type"]
            if parent and parent not in items:
                # Its parent is in the trash (or an annotation's attachment is): leave it out
                dropped.append(key)
                continue
            if kind == "attachment":
                mode = modes.get(d.get("linkMode"), LINKED_URL)
                path, p = None, d.get("path") or ""
                if mode in (IMPORTED_FILE, IMPORTED_URL) and d.get("filename"):
                    path = storage / key / d["filename"]
                elif mode == LINKED_FILE and p:
                    if p.startswith("attachments:"):
                        base_dir = base_dir or _base_attachment_dir()
                        path = base_dir / p[len("attachments:"):] if base_dir else None
                    else:
                        path = Path(p)
                info = {"key": key, "item_id": None, "title": d.get("title", ""), "link_mode": mode,
                        "content_type": d.get("contentType", ""), "path": path}
                owner = items[parent] if parent else it
                if parent:
                    items[parent]["attachments"].append(info)
                    it["_child"] = True
                else:
                    it["attachments"].append(info)       # standalone attachment: its own item
                # An attachment can carry a note of its own (the database keeps it in itemNotes)
                att_note = html_to_text(d.get("note", ""))
                if att_note:
                    owner["notes"].append({"key": key, "text": att_note})
            elif kind == "note":
                text = html_to_text(d.get("note", ""))
                entry = {"key": key, "text": text}
                if parent:
                    items[parent]["notes"].append(entry)
                    it["_child"] = True
                else:
                    first = text.split("\n", 1)[0].strip()
                    it["fields"].setdefault("title", first[:120] or "Untitled note")
                    it["notes"].append(entry)             # standalone note: its own item
            elif kind == "annotation":
                att = items[parent]["_data"]
                owner = att.get("parentItem") if att.get("parentItem") in items else parent
                items[owner]["annotations"].append({
                    "attachment_key": parent, "type": d.get("annotationType", "other"),
                    "text": (d.get("annotationText") or "").strip(),
                    "comment": (d.get("annotationComment") or "").strip(),
                    "page": d.get("annotationPageLabel") or "", "sort": d.get("annotationSortIndex", "")})
                it["_child"] = True
        for key in dropped:
            items.pop(key)
        for it in items.values():
            it.pop("_data", None)
        return _finish(items.values())


def open_source():
    """
    The local API if Zotero is running with it enabled, else a database snapshot.
    ZOTERO_SOURCE=api or =sqlite in .env forces one (api then fails if Zotero is closed).
    """
    choice = os.getenv("ZOTERO_SOURCE", "auto").lower()
    if choice in ("auto", "api"):
        try:
            src = ApiSource()
            log.info(f"Reading Zotero through the local API ({src.base})")
            return src
        except Exception as e:
            if choice == "api":
                raise RuntimeError(f"Zotero local API not reachable: {e}. Is Zotero running?") from e
            log.info(f"Zotero local API not reachable ({e.__class__.__name__}); using a database snapshot")
    return SqliteSource()


def find_library(libs: List[Dict], wanted: str) -> Optional[Dict]:
    """Match a library by slug, group ID or name (case-insensitive)."""
    w = wanted.strip().lower()
    for lib in libs:
        if w in (lib["slug"], str(lib["group_id"]), lib["name"].lower()):
            return lib
    return None


# ------------------------------------------------------------------ attachment text

def _clean_page(text: str) -> str:
    """Join hyphenated line breaks, drop lone page numbers, collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _extract(path: Path, content_type: str) -> List[Dict]:
    """[{page (label), index (1-based), text}] for one file; [] if it has no text."""
    suffix = path.suffix.lower()
    if content_type == "application/pdf" or suffix == ".pdf":
        import pymupdf
        pages = []
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PDF_PAGES:
                    break
                text = _clean_page(page.get_text())
                if text:
                    try:
                        label = page.get_label() or str(i + 1)   # printed page number, if the PDF has labels
                    except Exception:
                        label = str(i + 1)
                    pages.append({"page": label, "index": i + 1, "text": text})
        return pages
    if content_type.startswith("text/html") or suffix in (".html", ".htm"):
        text = html_to_text(path.read_text(errors="ignore"))
        return [{"page": "", "index": 0, "text": text}] if text else []
    if content_type.startswith("text/") or suffix in (".txt", ".md"):
        text = path.read_text(errors="ignore").strip()
        return [{"page": "", "index": 0, "text": text}] if text else []
    return []


def attachment_text(att: Dict, slug: str) -> Dict:
    """
    Text of one attachment, from the cache when the file hasn't changed.

    Falls back to Zotero's own full-text cache (.zotero-ft-cache, plain text
    without page numbers) for file types we can't read, e.g. Word documents,
    and for scanned PDFs that Zotero has OCR'd text for.

    Returns:
        {"status": "ok" | "missing" | "empty" | "error", "pages": [...]}. "missing"
        usually means the file hasn't synced down from Zotero storage yet.
    """
    path = att["path"]
    if att["link_mode"] == LINKED_URL or path is None:
        return {"status": "no-file", "pages": []}
    if not path.exists():
        return {"status": "missing", "pages": []}
    st = path.stat()
    sig = f"{st.st_size}:{int(st.st_mtime)}"
    cache_file = TEXT_CACHE / slug / f"{att['key']}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if cached.get("sig") == sig:
            return {"status": cached["status"], "pages": cached["pages"]}
    try:
        pages = _extract(path, att["content_type"])
        status = "ok"
    except Exception as e:
        log.warning(f"Could not read {path.name}: {e}")
        pages, status = [], "error"
    if not pages:
        ft = path.parent / ".zotero-ft-cache"
        if ft.exists() and ft.stat().st_size > 200:
            pages, status = [{"page": "", "index": 0, "text": _clean_page(ft.read_text(errors="ignore"))}], "ok"
        elif status == "ok":
            status = "empty"   # e.g. a scanned PDF without a text layer
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"sig": sig, "status": status, "pages": pages}))
    return {"status": status, "pages": pages}


def zotero_links(group_id: Optional[int], item_key: str, attachment_key: str = "",
                 page_index: int = 0) -> Dict[str, str]:
    """zotero:// links that select the item in Zotero and open the PDF at a page."""
    lib = f"groups/{group_id}" if group_id else "library"
    links = {"select": f"zotero://select/{lib}/items/{item_key}"}
    if attachment_key:
        links["open"] = f"zotero://open-pdf/{lib}/items/{attachment_key}" + (
            f"?page={page_index}" if page_index else "")
    return links


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for lib in open_source().list_libraries():
        print(f"{lib['slug']:18} {lib['items']:6}  {lib['name']}")
