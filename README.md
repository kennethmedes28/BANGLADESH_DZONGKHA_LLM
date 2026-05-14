# Voice RAG Pipeline

Audio or text query → speech-to-text (your adapter) → FAISS retrieval → OpenAI chat model → grounded answer with sources.

```
Audio File                    Text query
    │                              │
    ▼                              │
┌─────────────────────┐            │
│  VoiceToTextModel   │  ← Your pre-trained STT model (optional for text-only)
│  (custom adapter)   │            │
└─────────┬───────────┘            │
          │  transcript (str)      │
          └──────────┬─────────────┘
                     ▼
          ┌─────────────────────┐
          │   FAISS Retriever   │  ← Similarity search (top-k chunks)
          │  (LangChain wrapper)│
          └─────────┬───────────┘
                    │  relevant docs
                    ▼
          ┌─────────────────────┐
          │  OpenAI Chat model  │  ← Grounded answer (RetrievalQA)
          └─────────┬───────────┘
                    ▼
              Answer + Sources
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PDF ingestion, ensure `pypdf` is available (uncomment in `requirements.txt` or install with `pip install pypdf` if loaders fail).

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

On Windows (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`

For Streamlit only, you can instead set `OPENAI_API_KEY` in `.streamlit/secrets.toml`.

### 3. Add documents and build the FAISS index

Put PDFs and `.txt` files under `./docs/` (or another directory), then either:

```bash
python build_index.py --source ./docs --index faiss_index
```

or run the **index** stage via the orchestrator (see below).

### 4. Plug in your STT model

Open `pipeline.py` and implement `VoiceToTextModel`:

- `_load_model()` — load your trained model
- `transcribe(audio_path)` — return the transcript string

Until this is implemented, use **text-only** queries (`--text` or the Streamlit text path).

### 5. Run the pipeline

**Python API**

```python
from pipeline import VoiceRAGPipeline

pipe = VoiceRAGPipeline(
    stt_model_path="./models/my_stt_model",
    faiss_index_path="faiss_index",
)

result = pipe.run("query.wav")  # or pipe.query_text("your question")
print(result["answer"])
```

**CLI orchestration** (`orchestrate.py` runs stages in order: `index` → `voice_rag`)

```bash
python orchestrate.py list
```

Run the full pipeline (skips rebuilding the index if `faiss_index` already exists unless you pass `--rebuild-index`):

```bash
python orchestrate.py full --text "What is in the knowledge base?"
```

With audio (requires a working `VoiceToTextModel` and `--stt` or `STT_MODEL_PATH`):

```bash
python orchestrate.py full --audio ./samples/query.wav --stt ./models/my_stt_model
```

Run specific stages only:

```bash
python orchestrate.py steps index
python orchestrate.py steps voice_rag --text "Hello"
```

**Streamlit UI** (same stages as `orchestrate.py full`, with model picker and optional secrets):

```bash
streamlit run streamlit_app.py
```

## File structure

```
├── pipeline.py        # VoiceRAGPipeline: STT adapter + FAISS + RAG chain
├── build_index.py     # Chunk, embed, and save FAISS from ./docs (or --source)
├── orchestrate.py     # CLI: list / full / steps for index + voice_rag
├── streamlit_app.py   # Web UI to run stages with a chosen chat model
├── requirements.txt
└── README.md
```

Training notebooks and a `data/` directory may be present for your study; the deployed pipeline will depend on your best-performing model and layout.

## Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stt_model_path` | — | Path to your trained STT model (`--stt` / `STT_MODEL_PATH`) |
| `faiss_index_path` / `--index` | `faiss_index` | FAISS index directory |
| `llm_model` / `--llm` | `gpt-4o` | OpenAI chat model name |
| `top_k` / `--top-k` | `4` | Number of chunks to retrieve |
| `--chunk` | `500` | Chunk size for indexing |
| `--overlap` | `50` | Chunk overlap |

## Swapping the LLM

Pass `--llm` to `orchestrate.py`, set `llm_model` on `VoiceRAGPipeline`, or choose a preset in Streamlit. Examples: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`.

## Notes

- This repo is a **recipe** for a voice (or text) RAG stack: custom STT, LangChain + FAISS, OpenAI embeddings and chat. Wire your trained model into `VoiceToTextModel` when ready.
- **Orchestrator behavior:** `python orchestrate.py full` skips the index stage if the index path already exists; use `--rebuild-index` to force a rebuild.
- **Context:** All data needed for training may live under `data/` depending on your study; the final deployment pipeline can evolve with your best architecture.
