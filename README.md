# Voice RAG Pipeline

```
Audio File
    │
    ▼
┌─────────────────────┐
│  VoiceToTextModel   │  ← Your pre-trained STT model
│  (custom adapter)   │
└─────────┬───────────┘
          │  transcript (str)
          ▼
┌─────────────────────┐
│   FAISS Retriever   │  ← Similarity search (top-k chunks)
│  (LangChain wrapper)│
└─────────┬───────────┘
          │  relevant docs
          ▼
┌─────────────────────┐
│  OpenAI ChatGPT-4o  │  ← Generates grounded answer
│  (RetrievalQA chain)│
└─────────┬───────────┘
          │
          ▼
      Answer + Sources
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Build the FAISS index (one-time)
Put your knowledge base documents (PDFs, .txt) in `./docs/`, then:
```bash
python build_index.py --source ./docs --index faiss_index
```

### 4. Plug in your STT model
Open `pipeline.py` and edit `VoiceToTextModel`:
- `_load_model()` — load your trained model
- `transcribe(audio_path)` — return the transcript string

### 5. Run the pipeline
```python
from pipeline import VoiceRAGPipeline

pipe = VoiceRAGPipeline(
    stt_model_path="./models/my_stt_model",
    faiss_index_path="faiss_index",
)

result = pipe.run("query.wav")
print(result["answer"])
```

## File Structure
```
voice_rag_pipeline/
├── pipeline.py        # Main pipeline (STT + FAISS + RAG chain)
├── build_index.py     # One-time index builder
├── requirements.txt   # Python dependencies
└── README.md
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stt_model_path` | — | Path to your trained STT model |
| `faiss_index_path` | `faiss_index` | Directory where FAISS index is stored |
| `llm_model` | `gpt-4o` | OpenAI model name |
| `top_k` | `4` | Number of FAISS chunks to retrieve |
| `chunk_size` | `500` | Token size per chunk (build_index.py) |
| `chunk_overlap` | `50` | Overlap between chunks |

## Swapping the LLM
Change `llm_model` in `VoiceRAGPipeline` to any OpenAI model:
- `gpt-4o` (default — best quality)
- `gpt-4o-mini` (faster, cheaper)
- `gpt-3.5-turbo` (budget option)

## Notes
This project is a template for building a voice-based RAG system. This is the recipe for the pipeline architecture. Once the model has been trained, you can now setup the trained model in the `VoiceToTextModel` class and run the pipeline to get answers from your audio queries.

The API pipeline is streamlit-based, so you can easily deploy it as a web app. Just run:
```bash
streamlit run pipeline.py
```
This will launch a local web interface where you can upload audio files and see the generated answers along with the retrieved sources.