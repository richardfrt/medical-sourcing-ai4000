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

import os
import pandas as pd

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gudid_filtrado.csv")


def build_mock_dataset() -> pd.DataFrame:
    """Curated realistic GUDID-like sample.

    The first two rows are near-identical cardiovascular catheters:
    one premium (MedTronic-style) and one generic (B-Braun-style). This
    asymmetry is what the RAG engine exploits to surface savings.
    """

    rows = [
        {
            "deviceIdentifier": "00843123000011",
            "companyName": "CardioPremium Medical Inc.",
            "brandName": "ArteriaPro Elite",
            "versionModelNumber": "APE-6F-100",
            "gmdnPTName": "Cardiovascular catheter, percutaneous",
            "deviceDescription": (
                "Cateter cardiovascular percutaneo de 6Fr y 100 cm de longitud util, "
                "cuerpo de poliuretano reforzado con malla de acero inoxidable 316L, "
                "punta radiopaca de platino-iridio, recubrimiento hidrofilico para "
                "minima friccion en arteria femoral. Lumen interior de PTFE. "
                "Esterilizado por oxido de etileno. Compatible con guia 0.035 in."
            ),
        },
        {
            "deviceIdentifier": "08412345000028",
            "companyName": "GenericMed Solutions S.A.",
            "brandName": "VascuLine Standard",
            "versionModelNumber": "VLS-6F-100",
            "gmdnPTName": "Cardiovascular catheter, percutaneous",
            "deviceDescription": (
                "Cateter cardiovascular percutaneo 6Fr, longitud 100 cm, cuerpo de "
                "poliuretano con refuerzo de acero inoxidable 316L trenzado, punta "
                "radiopaca de platino-iridio, revestimiento hidrofilico de baja "
                "friccion. Lumen de PTFE. Esterilizacion por oxido de etileno. "
                "Compatible con guia estandar 0.035 in."
            ),
        },
        {
            "deviceIdentifier": "00847654000035",
            "companyName": "OrthoPremium GmbH",
            "brandName": "TitanScrew Pro",
            "versionModelNumber": "TSP-4.5-45",
            "gmdnPTName": "Orthopaedic bone screw, non-bioabsorbable",
            "deviceDescription": (
                "Tornillo ortopedico canulado de titanio grado 5 (Ti-6Al-4V), "
                "diametro 4.5 mm, longitud 45 mm, rosca autoperforante, cabeza "
                "hexalobular. Acabado anodizado tipo II. Esterilizacion terminal "
                "por radiacion gamma. Indicado para fijacion de fracturas metafisarias."
            ),
        },
        {
            "deviceIdentifier": "08411111000042",
            "companyName": "MedGenerics Iberia S.L.",
            "brandName": "BoneFix Basic",
            "versionModelNumber": "BFB-4.5-45",
            "gmdnPTName": "Orthopaedic bone screw, non-bioabsorbable",
            "deviceDescription": (
                "Tornillo ortopedico canulado fabricado en aleacion de titanio "
                "Ti-6Al-4V ELI, 4.5 mm de diametro por 45 mm de longitud, rosca "
                "autoperforante, recubrimiento anodizado. Esterilizado por gamma. "
                "Uso en osteosintesis metafisaria y diafisaria."
            ),
        },
        {
            "deviceIdentifier": "00849999000059",
            "companyName": "SutureMax Corp.",
            "brandName": "PolyKnot Premium",
            "versionModelNumber": "PKP-30-70",
            "gmdnPTName": "Suture, synthetic, absorbable",
            "deviceDescription": (
                "Sutura sintetica absorbible monofilamento de polidioxanona (PDS), "
                "calibre 3/0, longitud 70 cm, aguja cortante 3/8 de circulo de 26 mm. "
                "Absorcion en 180-210 dias. Esterilizada por oxido de etileno. "
                "Indicada para cierre de fascia y tejido subcutaneo."
            ),
        },
    ]

    return pd.DataFrame(rows, columns=[
        "deviceIdentifier",
        "companyName",
        "brandName",
        "versionModelNumber",
        "gmdnPTName",
        "deviceDescription",
    ])


def main() -> None:
    df = build_mock_dataset()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] CSV generado: {OUTPUT_PATH}")
    print(f"[OK] Filas: {len(df)} | Columnas: {list(df.columns)}")
    print(df[["brandName", "gmdnPTName"]].to_string(index=False))


if __name__ == "__main__":
    main()
