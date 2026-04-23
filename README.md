# MediSource AI — Inteligencia Clinica de Compras B2B

Motor SaaS de busqueda semantica (RAG) que identifica dispositivos medicos
genericos equivalentes a productos "de marca", y genera automaticamente un
**Informe de Justificacion Clinica** con GPT-4o para que el CFO y el jefe
de compras hospitalario puedan tomar decisiones seguras y rapidas.

## Arquitectura

- **Vector DB:** ChromaDB (`PersistentClient`)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** OpenAI `gpt-4o` (auditor clinico)
- **UI:** Streamlit (dashboard orientado a CFO)
- **Compatibilidad:** Local (Windows/Mac/Linux) + Streamlit Cloud (parche `pysqlite3`)

## Puesta en marcha

> **Requisito de Python:** Usa **Python 3.11** en local. Los pines del PRD
> (`chromadb==0.4.24`, `grpcio==1.62.1`, `protobuf==4.25.3`) son los que
> Streamlit Cloud resuelve sin colisiones y no publican wheels para
> Python 3.13/3.14. En Windows:
>
> ```powershell
> winget install Python.Python.3.11
> py -3.11 -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install --upgrade pip
> ```

1. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar la API key**

   Copia `.env.example` a `.env` y pega tu `OPENAI_API_KEY`.

3. **Generar el dataset mock**

   ```bash
   python generar_prueba.py
   ```

4. **Construir el indice vectorial**

   ```bash
   python gudid_embeddings.py
   ```

5. **Lanzar la app**

   ```bash
   streamlit run streamlit_app.py
   ```

## Despliegue en Streamlit Cloud

- Sube el repo a GitHub.
- En Streamlit Cloud anade el secret `OPENAI_API_KEY`.
- Entrypoint: `streamlit_app.py`.
- El parche `pysqlite3` al inicio de cada modulo hace que ChromaDB funcione
  en la nube sin tocar `sqlite3`.

## Ficheros

| Archivo | Rol |
|---|---|
| `generar_prueba.py` | Crea `gudid_filtrado.csv` con 5 dispositivos realistas. |
| `gudid_embeddings.py` | Indexa el CSV en ChromaDB con embeddings de OpenAI. |
| `logic_engine.py` | RAG: busca top-K + GPT-4o genera el informe. |
| `streamlit_app.py` | Dashboard B2B con KPIs y comparativa visual. |
