import importlib
from typing import List, Dict, Optional, Any

import streamlit as st

# "core-62" has a hyphen, so "from core-62 import ..." is a syntax error -- import it by its string name instead
run_llm = importlib.import_module("core-62").run_llm


def _format_context(context_docs: List[Any]) -> List[str]:
    """Return the source URL of each retrieved document."""
    return [
        str(meta.get("source") or "Unknown source")
        for doc in (context_docs or [])
        if (meta := getattr(doc, "metadata", None) or {}) is not None
    ]

st.set_page_config(page_title="LangChain Documentation RAG" ,layout="centered")
st.title("LangChain Documentation RAG")