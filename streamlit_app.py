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
import traceback

import streamlit as st

from logic_engine import AuditReport, DeviceHit, audit_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

st.set_page_config(
    page_title="MediSource AI | Inteligencia Clinica de Compras",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1rem;}
    .medisource-hero {
        background: linear-gradient(135deg, #0b3d91 0%, #1f7a8c 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .medisource-hero h1 {margin: 0; font-size: 1.9rem; color: white;}
    .medisource-hero p {margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 1rem;}
    .device-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        background: rgba(255, 255, 255, 0.02);
        height: 100%;
    }
    .device-card.premium {border-left: 5px solid #d62728;}
    .device-card.alt {border-left: 5px solid #2ca02c;}
    .device-card h4 {margin-top: 0;}
    .kpi-row {margin-top: 0.5rem;}
    .tag {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
        background: rgba(11, 61, 145, 0.12); color: #0b3d91;
        margin-right: 6px;
    }
    .tag.alt {background: rgba(44, 160, 44, 0.15); color: #1f7a3a;}
    .tag.premium {background: rgba(214, 39, 40, 0.12); color: #a31d1d;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
        <div class="medisource-hero">
            <h1>🏥 MediSource AI</h1>
            <p>Inteligencia clinica de compras B2B · Encuentra alternativas equivalentes y reduce el gasto hospitalario sin comprometer la seguridad clinica.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Configuracion")
        top_k = st.slider("Alternativas a considerar", min_value=2, max_value=5, value=3)
        st.divider()
        st.subheader("Ejemplos de busqueda")
        examples = [
            "Cateter cardiovascular percutaneo 6Fr",
            "Tornillo ortopedico canulado de titanio 4.5 mm",
            "Sutura sintetica absorbible 3/0",
        ]
        chosen_example = None
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                chosen_example = ex
        st.divider()
        st.caption("Motor: OpenAI `text-embedding-3-small` + `gpt-4o` · Vector DB: ChromaDB")
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("⚠️ Falta OPENAI_API_KEY en el entorno.")
        if not os.path.isdir(CHROMA_DIR):
            st.info(
                "📦 Indice vectorial no encontrado. "
                "Se inicializara automaticamente en la primera auditoria."
            )
    return {"top_k": top_k, "example": chosen_example}


def _render_device_card(
    title: str,
    hit: DeviceHit,
    price_label: str,
    style: str,
    tag_label: str,
) -> None:
    css_class = "device-card premium" if style == "premium" else "device-card alt"
    tag_class = "premium" if style == "premium" else "alt"
    st.markdown(
        f"""
        <div class="{css_class}">
            <span class="tag {tag_class}">{tag_label}</span>
            <h4>{title}</h4>
            <p style="margin: 0.2rem 0;"><b>{hit.brand_name}</b> &middot; {hit.company_name}</p>
            <p style="margin: 0.2rem 0; font-size: 0.88rem; opacity: 0.85;">
                Modelo <code>{hit.version_model_number}</code> · UDI <code>{hit.device_identifier}</code>
            </p>
            <p style="margin: 0.2rem 0; font-size: 0.88rem;">GMDN: <i>{hit.gmdn_pt_name}</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric(price_label, f"{hit.estimated_price:,.2f} €")
    with st.expander("Ficha tecnica"):
        st.write(hit.device_description)
        if hit.similarity:
            st.caption(f"Similitud semantica: {hit.similarity * 100:.1f}%")


def _render_results(report: AuditReport) -> None:
    if not report.anchor:
        st.warning("No se han encontrado dispositivos relevantes para esta busqueda.")
        return

    col_premium, col_alt = st.columns(2, gap="large")

    with col_premium:
        st.subheader("🔎 Producto encontrado")
        _render_device_card(
            title="Opcion premium actual",
            hit=report.anchor,
            price_label="Precio estimado (actual)",
            style="premium",
            tag_label="PREMIUM",
        )

    with col_alt:
        st.subheader("💡 Alternativa recomendada")
        if report.best_alternative:
            _render_device_card(
                title="Equivalente clinico mas eficiente",
                hit=report.best_alternative,
                price_label="Precio estimado (alternativa)",
                style="alt",
                tag_label="GENERICO EQUIVALENTE",
            )
        else:
            st.info("No hay alternativas con el mismo codigo GMDN en el catalogo actual.")

    st.divider()
    st.subheader("📊 Impacto financiero estimado")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Precio original", f"{report.anchor.estimated_price:,.2f} €")
    with k2:
        alt_price = (
            report.best_alternative.estimated_price if report.best_alternative else 0.0
        )
        st.metric("Precio alternativa", f"{alt_price:,.2f} €")
    with k3:
        st.metric(
            "Ahorro total",
            f"{report.savings_abs:,.2f} €",
            delta=f"-{report.savings_pct:.1f}%",
            delta_color="inverse",
        )

    st.divider()
    st.subheader("🧠 Informe de justificacion clinica (GPT-4o)")
    with st.expander("Ver informe completo del auditor clinico", expanded=True):
        if report.justification_markdown:
            st.markdown(report.justification_markdown)
        else:
            st.info("Sin informe disponible.")

    if len(report.alternatives) > 1:
        with st.expander("Otras alternativas encontradas"):
            for alt in report.alternatives[1:]:
                st.markdown(
                    f"- **{alt.brand_name}** ({alt.company_name}) — "
                    f"{alt.estimated_price:,.2f} € · similitud {alt.similarity * 100:.1f}%"
                )


def main() -> None:
    _render_hero()
    controls = _sidebar()

    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if controls["example"]:
        st.session_state["last_query"] = controls["example"]

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Busca un dispositivo medico",
            value=st.session_state["last_query"],
            placeholder="Ej: Cateter cardiovascular percutaneo 6Fr 100 cm",
        )
        submitted = st.form_submit_button(
            "🔍 Auditar gasto con IA", use_container_width=True, type="primary"
        )

    if submitted and query.strip():
        st.session_state["last_query"] = query
        try:
            with st.spinner("Analizando catalogo clinico y consultando GPT-4o..."):
                report = audit_search(query, top_k=controls["top_k"])
            _render_results(report)
        except Exception as exc:  # noqa: BLE001 - surface error to operator
            st.error(f"Error durante la auditoria: {exc}")
            with st.expander("Detalles tecnicos"):
                st.code(traceback.format_exc())
    elif submitted:
        st.warning("Introduce una descripcion del dispositivo para iniciar la auditoria.")


if __name__ == "__main__":
    main()
