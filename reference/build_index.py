"""
Step 3: chunks -> embeddings -> Chroma, with metadata from sources.csv.

Usage (from the project root, with .venv active):
    python -m ingest.build_index                     # build / update the index
    python -m ingest.build_index --rebuild           # delete and rebuild from scratch
    python -m ingest.build_index --test-query "annual fee for credit cards"

Outputs:
    chroma_db/                     the vector database (ignored by git)
    data/processed/chunks.jsonl    every chunk as plain text, so you can inspect them
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from ingest.chunk import make_chunks, split_sections
from ingest.parse import parse_pdf

SOURCES = Path("sources.csv")
DB_DIR = "chroma_db"
COLLECTION = "bank_docs"
CHUNKS_OUT = Path("data/processed/chunks.jsonl")

# Free, local, no API key. Downloads a small model (~80 MB) the first time.
# To switch models later, change this one line and run with --rebuild.
EMBEDDER = embedding_functions.DefaultEmbeddingFunction()

METADATA_FIELDS = ["bank", "country", "currency", "product_type", "document",
                   "url", "effective_date", "downloaded_on"]


def load_sources() -> list[dict]:
    with SOURCES.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    usable, skipped = [], []
    for row in rows:
        path = row.get("local_filename", "").strip()
        (usable if path and Path(path).exists() else skipped).append(row)
    for row in skipped:
        print(f"[skip] no file yet: {row['bank']} - {row['document']}")
    return usable


def embed_text(row: dict, section: str, text: str) -> str:
    """Prefix each chunk with where it came from, so a chunk that only says
    'Annual fee: 196.20' still matches a question about 'DBS credit cards'."""
    product = row["product_type"].replace("_", " ")
    return (f"{row['bank']} ({row['country']}, {row['currency']}) - {product} - "
            f"{row['document']}\nSection: {section}\n\n{text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="delete the index first")
    parser.add_argument("--test-query", help="run one search after building")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)
    if args.rebuild:
        try:
            client.delete_collection(COLLECTION)
            print("Deleted old index.")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        COLLECTION, embedding_function=EMBEDDER, metadata={"hnsw:space": "cosine"}
    )

    CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with CHUNKS_OUT.open("w", encoding="utf-8") as out:
        for row in load_sources():
            path = Path(row["local_filename"])
            lines = parse_pdf(path)
            if not lines:
                print(f"[warn] no text in {path} (scanned image? needs OCR) - skipped")
                continue

            chunks = make_chunks(split_sections(lines))
            ids, docs, metas = [], [], []
            for i, ch in enumerate(chunks):
                meta = {k: (row.get(k) or "") for k in METADATA_FIELDS}
                meta.update({
                    "source_file": str(path),
                    "section": ch.section[:500],
                    "page_start": ch.page_start,
                    "page_end": ch.page_end,
                    "chunk_index": i,
                })
                ids.append(f"{path.stem}::{i}")
                docs.append(embed_text(row, ch.section, ch.text))
                metas.append(meta)
                out.write(json.dumps({"id": ids[-1], "text": docs[-1], **meta},
                                     ensure_ascii=False) + "\n")

            # Remove this file's old chunks first, so re-running never leaves stale ones
            collection.delete(where={"source_file": str(path)})
            collection.add(ids=ids, documents=docs, metadatas=metas)
            total += len(chunks)
            print(f"[ok]   {path.name}: {len(lines)} lines -> {len(chunks)} chunks")

    print(f"\nIndexed {total} chunks. Collection now holds {collection.count()}.")
    print(f"Inspect them in {CHUNKS_OUT}")

    if args.test_query:
        res = collection.query(query_texts=[args.test_query], n_results=3)
        print(f'\nTop 3 results for "{args.test_query}":')
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            pages = (f"p.{meta['page_start']}" if meta["page_start"] == meta["page_end"]
                     else f"pp.{meta['page_start']}-{meta['page_end']}")
            print(f"\n- {meta['bank']} | {meta['document']} | {pages} | "
                  f"similarity {1 - dist:.2f}")
            print("  " + doc.split("\n\n", 1)[-1][:200].replace("\n", " ") + " ..." "test")


if __name__ == "__main__":
    main()