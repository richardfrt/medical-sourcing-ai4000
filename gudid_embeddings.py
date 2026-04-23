from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import hashlib
import os
import tempfile
import time
from typing import Iterable

import pandas as pd
from openai import OpenAI

import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "gudid_filtrado.csv")
COLLECTION_NAME = "medisource_devices"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 32


def _resolve_chroma_dir() -> str:
    env_dir = os.environ.get("CHROMA_DIR")
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir

    preferred_dir = os.path.join(BASE_DIR, "chroma_db")
    try:
        os.makedirs(preferred_dir, exist_ok=True)
        probe_path = os.path.join(preferred_dir, ".write_probe")
        with open(probe_path, "w", encoding="utf-8") as probe_file:
            probe_file.write("ok")
        os.remove(probe_path)
        return preferred_dir
    except Exception:
        fallback_dir = os.path.join(tempfile.gettempdir(), "medisource_chroma_db")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


CHROMA_DIR = _resolve_chroma_dir()


def _progress_bar(done: int, total: int, width: int = 32) -> str:
    ratio = 0 if total == 0 else done / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({ratio * 100:5.1f}%)"


def estimated_price(row: pd.Series) -> float:
    """Deterministic fake price.

    Premium brands (company/brand containing keywords like Premium, Pro,
    Elite) are anchored around ~120 EUR. Generic brands are anchored
    around ~40 EUR. A small hash-based jitter keeps prices distinct per
    device but stable across runs.
    """

    premium_markers = ("premium", "pro", "elite", "medtronic", "cardio")
    haystack = f"{row.get('companyName', '')} {row.get('brandName', '')}".lower()
    is_premium = any(marker in haystack for marker in premium_markers)

    seed = int(
        hashlib.md5(str(row.get("deviceIdentifier", "")).encode("utf-8")).hexdigest(),
        16,
    )
    jitter = (seed % 1000) / 100.0  # 0.00 .. 9.99

    base = 120.0 if is_premium else 40.0
    return round(base + jitter, 2)


def _chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def build_index(reset: bool = True) -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"No se encontro {CSV_PATH}. Ejecuta primero: python generar_prueba.py"
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY en el entorno. Crea un .env con OPENAI_API_KEY=sk-..."
        )

    print(f"[INFO] Leyendo CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).fillna("")
    df["estimated_price"] = df.apply(estimated_price, axis=1)
    total = len(df)
    print(f"[INFO] {total} dispositivos cargados.")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"[INFO] Coleccion previa '{COLLECTION_NAME}' eliminada.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    client = OpenAI()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for _, row in df.iterrows():
        ids.append(str(row["deviceIdentifier"]))
        documents.append(str(row["deviceDescription"]))
        metadatas.append(
            {
                "deviceIdentifier": str(row["deviceIdentifier"]),
                "companyName": str(row["companyName"]),
                "brandName": str(row["brandName"]),
                "versionModelNumber": str(row["versionModelNumber"]),
                "gmdnPTName": str(row["gmdnPTName"]),
                "deviceDescription": str(row["deviceDescription"]),
                "estimated_price": float(row["estimated_price"]),
            }
        )

    print("[INFO] Vectorizando descripciones con OpenAI...")
    done = 0
    t0 = time.time()
    for batch_docs, batch_ids, batch_meta in zip(
        _chunked(documents, BATCH_SIZE),
        _chunked(ids, BATCH_SIZE),
        _chunked(metadatas, BATCH_SIZE),
    ):
        embeddings = embed_batch(client, batch_docs)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
        )
        done += len(batch_docs)
        print(_progress_bar(done, total), end="\r", flush=True)

    elapsed = time.time() - t0
    print()
    print(f"[OK] Indexados {done} documentos en {elapsed:.1f}s")
    print(f"[OK] ChromaDB persistido en: {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
