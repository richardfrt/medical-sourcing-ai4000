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

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "medisource_devices"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "Eres un Auditor Clinico de Compras. Compara el producto buscado con las "
    "alternativas. Redacta un informe breve y profesional justificando si la "
    "sustitucion es segura basandote en los materiales y el codigo GMDN. "
    "Confirma el % de ahorro."
)


@dataclass
class DeviceHit:
    device_identifier: str
    company_name: str
    brand_name: str
    version_model_number: str
    gmdn_pt_name: str
    device_description: str
    estimated_price: float
    similarity: float = 0.0

    @classmethod
    def from_metadata(cls, meta: dict, distance: float) -> "DeviceHit":
        similarity = max(0.0, 1.0 - float(distance))
        return cls(
            device_identifier=str(meta.get("deviceIdentifier", "")),
            company_name=str(meta.get("companyName", "")),
            brand_name=str(meta.get("brandName", "")),
            version_model_number=str(meta.get("versionModelNumber", "")),
            gmdn_pt_name=str(meta.get("gmdnPTName", "")),
            device_description=str(meta.get("deviceDescription", "")),
            estimated_price=float(meta.get("estimated_price", 0.0)),
            similarity=round(similarity, 4),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    query: str
    anchor: DeviceHit | None
    alternatives: list[DeviceHit] = field(default_factory=list)
    best_alternative: DeviceHit | None = None
    savings_abs: float = 0.0
    savings_pct: float = 0.0
    justification_markdown: str = ""


def _get_openai_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Configuralo en tu .env o en los Secrets de Streamlit Cloud."
        )
    return OpenAI()


def _get_collection():
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except InvalidCollectionException:
        _bootstrap_collection_if_missing()
        try:
            return chroma_client.get_collection(COLLECTION_NAME)
        except InvalidCollectionException as inner_exc:
            raise RuntimeError(
                "No se pudo inicializar la coleccion vectorial 'medisource_devices'. "
                "Verifica OPENAI_API_KEY y vuelve a intentar."
            ) from inner_exc
        except Exception as inner_exc:  # noqa: BLE001
            raise RuntimeError(
                "Fallo al cargar la coleccion vectorial despues del bootstrap automatico."
            ) from inner_exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("No se pudo abrir la base vectorial de ChromaDB.") from exc


def _bootstrap_collection_if_missing() -> None:
    """Create demo CSV + Chroma collection when first run has no index."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Configuralo en Secrets para crear el indice vectorial."
        )

    from generar_prueba import OUTPUT_PATH, build_mock_dataset
    from gudid_embeddings import build_index

    if not os.path.exists(OUTPUT_PATH):
        df = build_mock_dataset()
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    build_index(reset=False)


def embed_query(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def semantic_search(query: str, top_k: int = 3) -> list[DeviceHit]:
    """Return top_k devices ranked by cosine similarity to the query."""

    client = _get_openai_client()
    collection = _get_collection()

    embedding = embed_query(client, query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    hits: list[DeviceHit] = []
    for meta, distance in zip(metadatas, distances):
        hits.append(DeviceHit.from_metadata(meta, distance))
    return hits


def _pick_anchor_and_alternatives(
    hits: list[DeviceHit],
) -> tuple[DeviceHit | None, list[DeviceHit]]:
    """The anchor is the most expensive match (the "premium" product the
    buyer probably typed). Alternatives are the remaining cheaper hits
    with the same GMDN code."""

    if not hits:
        return None, []

    anchor = max(hits, key=lambda h: h.estimated_price)
    alternatives = [
        h
        for h in hits
        if h.device_identifier != anchor.device_identifier
        and h.gmdn_pt_name == anchor.gmdn_pt_name
    ]
    alternatives.sort(key=lambda h: h.estimated_price)
    return anchor, alternatives


def _build_user_prompt(anchor: DeviceHit, alternatives: list[DeviceHit]) -> str:
    payload = {
        "producto_buscado": anchor.to_dict(),
        "alternativas": [a.to_dict() for a in alternatives],
    }
    return (
        "Analiza los siguientes dispositivos medicos y entrega tu informe en "
        "Markdown con estas secciones: \n"
        "1. **Resumen ejecutivo** (2 frases)\n"
        "2. **Equivalencia tecnica** (materiales, GMDN, especificaciones)\n"
        "3. **Riesgos clinicos** (si los hay)\n"
        "4. **Recomendacion y ahorro** (incluye % de ahorro calculado)\n\n"
        f"Datos:\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


def generate_justification(anchor: DeviceHit, alternatives: list[DeviceHit]) -> str:
    if not alternatives:
        return (
            "_No se han encontrado alternativas con el mismo codigo GMDN. "
            "Sugerencia: ampliar el catalogo o revisar la consulta._"
        )

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(anchor, alternatives)},
        ],
    )
    return response.choices[0].message.content or ""


def audit_search(query: str, top_k: int = 3) -> AuditReport:
    """End-to-end pipeline: search + pick anchor + justify with GPT-4o."""

    hits = semantic_search(query, top_k=top_k)
    anchor, alternatives = _pick_anchor_and_alternatives(hits)

    report = AuditReport(query=query, anchor=anchor, alternatives=alternatives)

    if anchor and alternatives:
        best = alternatives[0]
        report.best_alternative = best
        report.savings_abs = round(anchor.estimated_price - best.estimated_price, 2)
        if anchor.estimated_price > 0:
            report.savings_pct = round(
                100.0 * report.savings_abs / anchor.estimated_price, 1
            )

    report.justification_markdown = (
        generate_justification(anchor, alternatives) if anchor else ""
    )
    return report


if __name__ == "__main__":
    import pprint

    demo = audit_search("cateter cardiovascular percutaneo 6Fr")
    pprint.pp(demo)
