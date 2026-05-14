"""
Voice-to-Text → RAG Pipeline
Stack: Custom STT model | OpenAI LLM | FAISS Vector DB | LangChain
"""

import os
from pathlib import Path
from typing import Optional

# LangChain core
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ─────────────────────────────────────────────
# 1. VOICE-TO-TEXT WRAPPER
# ─────────────────────────────────────────────

class VoiceToTextModel:
    """
    Adapter for your pre-trained STT model.
    Replace the body of `transcribe()` with your actual model's inference call.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Load your trained STT model here."""
        # ── Example for Whisper-based model ──────────────────────────
        # import whisper
        # return whisper.load_model(self.model_path)
        #
        # ── Example for a HuggingFace model ──────────────────────────
        # from transformers import pipeline
        # return pipeline("automatic-speech-recognition", model=self.model_path)
        #
        # ── Replace with your actual loader ──────────────────────────
        raise NotImplementedError(
            f"Load your STT model from: {self.model_path}\n"
            "Uncomment the appropriate block above."
        )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file and return plain text.
        Replace with your model's inference API.
        """
        # ── Whisper ───────────────────────────────────────────────────
        # result = self.model.transcribe(audio_path)
        # return result["text"].strip()
        #
        # ── HuggingFace ASR pipeline ──────────────────────────────────
        # result = self.model(audio_path)
        # return result["text"].strip()
        raise NotImplementedError("Implement transcribe() for your model.")


# ─────────────────────────────────────────────
# 2. FAISS VECTOR STORE MANAGER
# ─────────────────────────────────────────────

class FAISSManager:
    """Build, persist, and query a FAISS index via LangChain."""

    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",       # swap to 'large' for better recall
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        self.vectorstore: Optional[FAISS] = None

    # ── Build ──────────────────────────────────────────────────────────
    def build_from_texts(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """Create a new FAISS index from a list of plain strings."""
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.save()
        print(f"✅ FAISS index built with {len(docs)} documents → saved to '{self.index_path}'")

    def build_from_documents(self, docs: list[Document]) -> None:
        """Create a new FAISS index from LangChain Document objects."""
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.save()

    # ── Persist ────────────────────────────────────────────────────────
    def save(self) -> None:
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)

    def load(self) -> None:
        """Load an existing FAISS index from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"No FAISS index found at '{self.index_path}'. "
                "Call build_from_texts() first."
            )
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"✅ FAISS index loaded from '{self.index_path}'")

    # ── Add new docs at runtime ────────────────────────────────────────
    def add_texts(self, texts: list[str], metadatas: list[dict] = None) -> None:
        if not self.vectorstore:
            raise RuntimeError("Load or build the index first.")
        self.vectorstore.add_texts(texts, metadatas=metadatas or [{}] * len(texts))
        self.save()

    # ── Retriever ──────────────────────────────────────────────────────
    def as_retriever(self, k: int = 4):
        if not self.vectorstore:
            raise RuntimeError("Load or build the index first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# ─────────────────────────────────────────────
# 3. RAG CHAIN BUILDER
# ─────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY
the context provided below. If the answer is not in the context, say
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
)


def build_rag_chain(retriever, model: str = "gpt-4o", temperature: float = 0.2) -> RetrievalQA:
    """Return a RetrievalQA chain wired to an OpenAI LLM."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                # use 'map_reduce' for very long contexts
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


# ─────────────────────────────────────────────
# 4. FULL PIPELINE
# ─────────────────────────────────────────────

class VoiceRAGPipeline:
    """
    End-to-end pipeline:
      audio file → STT → query → FAISS retrieval → OpenAI LLM → answer
    """

    def __init__(
        self,
        stt_model_path: Optional[str] = None,
        faiss_index_path: str = "faiss_index",
        llm_model: str = "gpt-4o",
        top_k: int = 4,
    ):
        print("🔧 Initialising Voice RAG Pipeline …")

        # STT (optional — omit path to use query_text() only until STT is wired)
        self.stt: Optional[VoiceToTextModel]
        if stt_model_path:
            self.stt = VoiceToTextModel(stt_model_path)
        else:
            self.stt = None
            print("   (STT disabled — use query_text() or pass stt_model_path for audio.run())")

        # FAISS
        self.faiss_mgr = FAISSManager(index_path=faiss_index_path)
        self.faiss_mgr.load()                       # loads existing index

        # RAG chain
        retriever = self.faiss_mgr.as_retriever(k=top_k)
        self.rag_chain = build_rag_chain(retriever, model=llm_model)

        print("✅ Pipeline ready.\n")

    # ── Main entry point ───────────────────────────────────────────────
    def run(self, audio_path: str) -> dict:
        """
        Args:
            audio_path: Path to the audio file (.wav / .mp3 / etc.)
        Returns:
            {
              "transcript": str,
              "answer":     str,
              "sources":    list[Document],
            }
        """
        if not self.stt:
            raise RuntimeError(
                "Audio run requires STT. Pass stt_model_path to VoiceRAGPipeline, "
                "or use query_text() for text-only queries."
            )
        print(f"🎙️  Transcribing: {audio_path}")
        transcript = self.stt.transcribe(audio_path)
        print(f"📝  Transcript  : {transcript}\n")

        print("🔍  Retrieving context from FAISS …")
        result = self.rag_chain.invoke({"query": transcript})

        answer = result["result"]
        sources = result.get("source_documents", [])

        print(f"💬  Answer      : {answer}\n")
        if sources:
            print("📚  Sources used:")
            for i, doc in enumerate(sources, 1):
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"   [{i}] {preview} …")

        return {"transcript": transcript, "answer": answer, "sources": sources}

    # ── Convenience: query with raw text (no audio) ────────────────────
    def query_text(self, text: str) -> dict:
        result = self.rag_chain.invoke({"query": text})
        return {
            "transcript": text,
            "answer": result["result"],
            "sources": result.get("source_documents", []),
        }


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── One-time: build the FAISS index from your knowledge base ───────
    # from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    #
    # loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    # raw_docs = loader.load()
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = splitter.split_documents(raw_docs)
    #
    # mgr = FAISSManager("faiss_index")
    # mgr.build_from_documents(chunks)
    # ───────────────────────────────────────────────────────────────────

    pipeline = VoiceRAGPipeline(
        stt_model_path="./models/my_stt_model",   # ← your model path
        faiss_index_path="faiss_index",
        llm_model="gpt-4o",
        top_k=4,
    )

    # Run on an audio file
    output = pipeline.run("sample_query.wav")

    # Or query with text directly (for testing without audio)
    # output = pipeline.query_text("What is the refund policy?")
