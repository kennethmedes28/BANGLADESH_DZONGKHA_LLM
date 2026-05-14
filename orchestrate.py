"""
Single entry point to run every pipeline stage defined in this project, in order.

Registered stages (see PIPELINE_ORDER):
  index      — embed documents into FAISS (wraps build_index.build)
  voice_rag  — load retriever + LLM and answer from text or audio (VoiceRAGPipeline)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Sequence

# Ordered stages used by the `full` command and by explicit `--steps`.
PIPELINE_ORDER: tuple[str, ...] = ("index", "voice_rag")

PIPELINE_HELP: dict[str, str] = {
    "index": "Load PDFs/text from --source, chunk, embed, save FAISS to --index.",
    "voice_rag": "Load FAISS + LLM; transcribe --audio with STT or use --text.",
}


def _require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running this command.")


def stage_index(source: str, index: str, chunk: int, overlap: int) -> None:
    from build_index import build as build_faiss_index

    _require_openai_key()
    print("\n── Stage: index ──")
    build_faiss_index(source, index, chunk, overlap)


def stage_voice_rag(
    index: str,
    llm_model: str,
    top_k: int,
    stt_model_path: str | None,
    text: str | None,
    audio: str | None,
) -> dict:
    from pipeline import VoiceRAGPipeline

    _require_openai_key()
    print("\n── Stage: voice_rag ──")
    pipe = VoiceRAGPipeline(
        stt_model_path=stt_model_path,
        faiss_index_path=index,
        llm_model=llm_model,
        top_k=top_k,
    )
    out: dict | None = None
    if text:
        out = pipe.query_text(text)
        print(f"💬  Answer: {out['answer']}\n")
    if audio:
        out = pipe.run(audio)
    assert out is not None
    return out


def run_selected_steps(
    steps: Sequence[str],
    *,
    source: str,
    index: str,
    chunk: int,
    overlap: int,
    llm_model: str,
    top_k: int,
    stt_model_path: str | None,
    text: str | None,
    audio: str | None,
    skip_index_if_exists: bool,
) -> dict | None:
    unknown = [s for s in steps if s not in PIPELINE_ORDER]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}. Valid: {list(PIPELINE_ORDER)}")

    result: dict | None = None
    for step in steps:
        if step == "index":
            if skip_index_if_exists and Path(index).exists():
                print(f"\n── Stage: index (skipped; '{index}' already exists) ──")
                continue
            stage_index(source, index, chunk, overlap)
        elif step == "voice_rag":
            if not text and not audio:
                print(
                    "\n── Stage: voice_rag (skipped; provide --text and/or --audio) ──"
                )
                continue
            result = stage_voice_rag(
                index=index,
                llm_model=llm_model,
                top_k=top_k,
                stt_model_path=stt_model_path,
                text=text,
                audio=audio,
            )
    return result


def _add_common_index_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="./docs", help="Document root for index stage")
    p.add_argument("--index", default="faiss_index", help="FAISS output directory")
    p.add_argument("--chunk", type=int, default=500, help="Chunk size for index stage")
    p.add_argument("--overlap", type=int, default=50, help="Chunk overlap for index stage")


def _add_common_rag_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", dest="llm_model", default="gpt-4o", help="OpenAI chat model name")
    p.add_argument("--top-k", type=int, default=4, dest="top_k", help="Retriever top-k")
    p.add_argument(
        "--stt",
        dest="stt_model_path",
        default=os.environ.get("STT_MODEL_PATH"),
        help="STT model path (or set STT_MODEL_PATH). Omit when using only --text.",
    )
    p.add_argument("--text", default=None, help="Query string (text-only RAG)")
    p.add_argument("--audio", default=None, help="Audio file path (requires --stt)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate in-repo pipelines (index build + voice RAG) in one run.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Print registered pipeline stages and order")
    p_list.set_defaults(func=_cmd_list)

    p_full = sub.add_parser("full", help=f"Run stages in order: {' → '.join(PIPELINE_ORDER)}")
    _add_common_index_args(p_full)
    _add_common_rag_args(p_full)
    p_full.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Always run index stage even if --index already exists",
    )
    p_full.set_defaults(func=_cmd_full)

    p_steps = sub.add_parser("steps", help="Run an explicit subset/order of stage names")
    p_steps.add_argument(
        "names",
        nargs="+",
        choices=PIPELINE_ORDER,
        help="Stage names to run (e.g. index voice_rag)",
    )
    _add_common_index_args(p_steps)
    _add_common_rag_args(p_steps)
    p_steps.add_argument(
        "--rebuild-index",
        action="store_true",
        help="When 'index' is included, run it even if --index already exists",
    )
    p_steps.set_defaults(func=_cmd_steps)

    args = parser.parse_args()
    func: Callable[[argparse.Namespace], None] = args.func
    func(args)


def _cmd_list(_: argparse.Namespace) -> None:
    print("Pipeline stages (in default orchestration order):\n")
    for i, name in enumerate(PIPELINE_ORDER, 1):
        print(f"  {i}. {name}\n     {PIPELINE_HELP[name]}\n")


def _cmd_full(args: argparse.Namespace) -> None:
    skip = not args.rebuild_index
    run_selected_steps(
        PIPELINE_ORDER,
        source=args.source,
        index=args.index,
        chunk=args.chunk,
        overlap=args.overlap,
        llm_model=args.llm_model,
        top_k=args.top_k,
        stt_model_path=args.stt_model_path,
        text=args.text,
        audio=args.audio,
        skip_index_if_exists=skip,
    )


def _cmd_steps(args: argparse.Namespace) -> None:
    skip = not args.rebuild_index
    run_selected_steps(
        args.names,
        source=args.source,
        index=args.index,
        chunk=args.chunk,
        overlap=args.overlap,
        llm_model=args.llm_model,
        top_k=args.top_k,
        stt_model_path=args.stt_model_path,
        text=args.text,
        audio=args.audio,
        skip_index_if_exists=skip,
    )


if __name__ == "__main__":
    main()
