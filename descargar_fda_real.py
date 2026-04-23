from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://accessgudid.nlm.nih.gov/api/v2/devices/search.json"
FALLBACK_API_URL = "https://accessgudid.nlm.nih.gov/api/v2/devices/implantable/list.json"
SEARCH_TERM = "catheter"
TARGET_RECORDS = 300
PAGE_SIZE = 100
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gudid_real.csv")
TIMEOUT_SECONDS = 45


def _build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _first_text(obj: Any, candidate_keys: list[str]) -> str:
    if isinstance(obj, dict):
        for key in candidate_keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_devices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = [
        payload.get("results"),
        payload.get("devices"),
        payload.get("device"),
    ]

    results = payload.get("results")
    if isinstance(results, dict):
        candidates.extend(
            [
                results.get("devices"),
                results.get("device"),
                results.get("content"),
                results.get("results"),
            ]
        )

    for item in candidates:
        if isinstance(item, list):
            return [x for x in item if isinstance(x, dict)]
        if isinstance(item, dict):
            nested_lists = [v for v in item.values() if isinstance(v, list)]
            for nested in nested_lists:
                dict_items = [x for x in nested if isinstance(x, dict)]
                if dict_items:
                    return dict_items

    return []


def _map_identifier(device: dict[str, Any]) -> str:
    direct = _first_text(device, ["deviceIdentifier", "identifier"])
    if direct:
        return direct

    for identifier in _as_list(device.get("identifiers")):
        if not isinstance(identifier, dict):
            continue
        value = _first_text(
            identifier,
            [
                "deviceIdentifier",
                "identifier",
                "id",
                "value",
            ],
        )
        if value:
            return value
    return ""


def _map_gmdn(device: dict[str, Any]) -> str:
    direct = _first_text(device, ["gmdnPTName", "gmdnName"])
    if direct:
        return direct

    gmdn_pt_name = device.get("gmdnPTName")
    if isinstance(gmdn_pt_name, list):
        for item in gmdn_pt_name:
            if isinstance(item, str) and item.strip():
                return item.strip()

    for term in _as_list(device.get("gmdnTerms")):
        if not isinstance(term, dict):
            continue
        value = _first_text(term, ["gmdnPTName", "gmdnName", "termName"])
        if value:
            return value
    return ""


def _map_device(device: dict[str, Any]) -> dict[str, str]:
    return {
        "deviceIdentifier": _map_identifier(device),
        "companyName": _first_text(device, ["companyName", "labelerName"]),
        "brandName": _first_text(device, ["brandName", "deviceName"]),
        "versionModelNumber": _first_text(
            device, ["versionModelNumber", "versionOrModelNumber"]
        ),
        "gmdnPTName": _map_gmdn(device),
        "deviceDescription": _first_text(
            device,
            ["deviceDescription", "description", "deviceRecordStatus"],
        ),
    }


def _request_page(
    session: requests.Session, page: int, page_size: int, search_term: str
) -> tuple[dict[str, Any], bool]:
    param_attempts = [
        {"search": search_term, "page": page, "limit": page_size},
        {"q": search_term, "page": page, "limit": page_size},
    ]

    last_error: Exception | None = None
    for params in param_attempts:
        try:
            response = session.get(API_URL, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if _extract_devices(payload):
                return payload, False
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    fallback_response = session.get(
        FALLBACK_API_URL,
        params={"page": page, "per_page": page_size},
        timeout=TIMEOUT_SECONDS,
    )
    fallback_response.raise_for_status()
    return fallback_response.json(), True


def download_real_gudid(
    target_records: int = TARGET_RECORDS,
    search_term: str = SEARCH_TERM,
    page_size: int = PAGE_SIZE,
) -> pd.DataFrame:
    session = _build_session()
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    page = 1
    fallback_scan_pages = 10

    while len(rows) < target_records:
        payload, using_fallback = _request_page(
            session=session, page=page, page_size=page_size, search_term=search_term
        )
        devices = _extract_devices(payload)
        if not devices:
            print(f"[WARN] Sin resultados en pagina {page}.")
            break

        page_added = 0
        for device in devices:
            if using_fallback:
                haystack = " ".join(
                    [
                        str(device.get("brandName", "")),
                        str(device.get("companyName", "")),
                        str(device.get("versionModelNumber", "")),
                        str(device.get("deviceDescription", "")),
                        str(device.get("gmdnPTName", "")),
                    ]
                ).lower()
                keep_unfiltered = page > fallback_scan_pages
                if search_term.lower() not in haystack and not keep_unfiltered:
                    continue

            mapped = _map_device(device)
            device_id = mapped.get("deviceIdentifier", "").strip()
            dedup_key = device_id or f"fallback-{page}-{page_added}"
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)
            rows.append(mapped)
            page_added += 1
            if len(rows) >= target_records:
                break

        mode = "fallback-implantable" if using_fallback else "search"
        print(
            f"[INFO] Pagina {page} ({mode}): +{page_added} "
            f"(total {len(rows)}/{target_records})"
        )
        page += 1
        if len(rows) < target_records:
            time.sleep(1)

    df = pd.DataFrame(
        rows[:target_records],
        columns=[
            "deviceIdentifier",
            "companyName",
            "brandName",
            "versionModelNumber",
            "gmdnPTName",
            "deviceDescription",
        ],
    ).fillna("")
    return df


def main() -> None:
    print("[INFO] Descargando datos reales de AccessGUDID...")
    df = download_real_gudid()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] CSV real generado: {OUTPUT_PATH}")
    print(f"[OK] Registros guardados: {len(df)}")


if __name__ == "__main__":
    main()
