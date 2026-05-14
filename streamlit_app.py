"""
Streamlit UI to run all orchestrated pipeline stages with a chosen OpenAI chat model.

Stages match `orchestrate.PIPELINE_ORDER`: index → voice_rag.

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

import streamlit as st

from orchestrate import PIPELINE_HELP, PIPELINE_ORDER, run_selected_steps

# Common OpenAI chat model presets; user can override with custom text.
LLM_PRESETS: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o4-mini",
    "o3-mini",
    "o1-mini",
    "o1",
)


def _resolve_llm_model(preset: str, custom: str) -> str:
    if preset == "Custom":
        return (custom or "").strip() or "gpt-4o"
    return preset


def _ensure_openai_key_from_secrets() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
    except (FileNotFoundError, KeyError, RuntimeError, TypeError):
        return


def main() -> None:
    st.set_page_config(page_title="Voice RAG — Pipeline", layout="wide")
    st.title("Voice RAG pipeline")
    st.caption(
        "Run indexing and voice/text RAG in order. Stages are the same as `python orchestrate.py full`."
    )

    _ensure_openai_key_from_secrets()
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "Set **OPENAI_API_KEY** in your environment or in `.streamlit/secrets.toml` "
            "(`OPENAI_API_KEY = \"sk-...\"`) before running stages."
        )
        st.stop()

    with st.sidebar:
        st.header("Model")
        llm_preset = st.selectbox("Chat model", options=(*LLM_PRESETS, "Custom"), index=0)
        llm_custom = st.text_input(
            "Custom model id",
            placeholder="e.g. gpt-4.1",
            disabled=llm_preset != "Custom",
        )
        llm_model = _resolve_llm_model(llm_preset, llm_custom)

        st.header("Stages to run")
        run_all = st.checkbox("Run all in order", value=True, help=str(list(PIPELINE_ORDER)))
        if run_all:
            selected = list(PIPELINE_ORDER)
            for name in PIPELINE_ORDER:
                st.markdown(f"- **{name}** — {PIPELINE_HELP[name]}")
        else:
            selected = [
                s
                for s in PIPELINE_ORDER
                if st.checkbox(s, value=True, help=PIPELINE_HELP.get(s, ""))
            ]

        st.header("Index stage")
        source = st.text_input("Document root (`--source`)", value="./docs")
        index_dir = st.text_input("FAISS output (`--index`)", value="faiss_index")
        chunk = st.number_input("Chunk size", min_value=50, value=500, step=50)
        overlap = st.number_input("Chunk overlap", min_value=0, value=50, step=10)
        rebuild_index = st.checkbox(
            "Always rebuild index",
            value=False,
            help="If off, index step is skipped when the index path already exists.",
        )

        st.header("RAG stage")
        top_k = st.number_input("Retriever top-k", min_value=1, value=4, step=1)
        stt_path = st.text_input(
            "STT model path (optional)",
            value=os.environ.get("STT_MODEL_PATH") or "",
            help="Required for audio queries until `VoiceToTextModel` is implemented in pipeline.py.",
        )
        stt_model_path = stt_path.strip() or None

    col1, col2 = st.columns(2)
    with col1:
        query_text = st.text_area("Text query (optional)", height=120, placeholder="Ask in plain text …")
    with col2:
        audio_file = st.file_uploader(
            "Audio file (optional)",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )

    text_arg = query_text.strip() or None
    audio_path: str | None = None
    if audio_file is not None:
        suffix = Path(audio_file.name).suffix or ".wav"
        fd, audio_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        Path(audio_path).write_bytes(audio_file.getvalue())

    run_btn = st.button("Run selected stages", type="primary")

    if run_btn:
        if not selected:
            st.warning("Select at least one stage in the sidebar.")
        elif "voice_rag" in selected and not text_arg and not audio_path:
            st.warning(
                "The **voice_rag** stage is selected but there is no text query or audio file."
            )
        else:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    result = run_selected_steps(
                        selected,
                        source=source,
                        index=index_dir,
                        chunk=int(chunk),
                        overlap=int(overlap),
                        llm_model=llm_model,
                        top_k=int(top_k),
                        stt_model_path=stt_model_path,
                        text=text_arg,
                        audio=audio_path,
                        skip_index_if_exists=not rebuild_index,
                    )
            except Exception as e:
                st.exception(e)
                st.code(buf.getvalue() or "(no log output)", language="text")
            else:
                st.success("Finished.")
                with st.expander("Console log", expanded=False):
                    st.code(buf.getvalue() or "(no log output)", language="text")
                if result is not None:
                    st.subheader("RAG output")
                    st.markdown(f"**Model:** `{llm_model}`")
                    if result.get("transcript"):
                        st.markdown("**Transcript / query**")
                        st.write(result["transcript"])
                    st.markdown("**Answer**")
                    st.write(result.get("answer", ""))
                    sources = result.get("sources") or []
                    if sources:
                        with st.expander("Source chunks", expanded=False):
                            for i, doc in enumerate(sources, 1):
                                meta = getattr(doc, "metadata", {}) or {}
                                st.markdown(f"**[{i}]** `{meta}`")
                                st.text((doc.page_content or "")[:2000])
            finally:
                if audio_path and Path(audio_path).exists():
                    try:
                        Path(audio_path).unlink(missing_ok=True)
                    except OSError:
                        pass


if __name__ == "__main__":
    main()
