import os, httpx
from dotenv import load_dotenv
load_dotenv()

# API Configuration (https://api.hcnsec.cn/)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.hcnsec.cn/v1").rstrip("/")
API_KEY = os.getenv("API_KEY", os.getenv("HCNSEC_API_KEY", "your_hcnsec_api_key"))
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Any, List, Optional


class HcnsecLLM(BaseChatModel):
    model: str = MODEL_NAME
    temperature: float = 0.3

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        import time
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user" if m.type == "human" else m.type, "content": m.content} for m in messages],
        }
        for attempt in range(5):
            r = httpx.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        raise Exception("API rate limit exceeded after retries. Please wait a moment and try again.")

    @property
    def _llm_type(self) -> str:
        return "hcnsec"


OpenRouterLLM = HcnsecLLM  # Backward compatibility alias

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

llm = HcnsecLLM()
vector_stores: dict[str, FAISS] = {}


def ingest_pdf(session_id: str, file_path: str):
    docs = PyPDFLoader(file_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    vector_stores[session_id] = FAISS.from_documents(chunks, embeddings)


def get_retriever(session_id: str):
    if session_id not in vector_stores:
        raise ValueError("No PDF uploaded for this session.")
    return vector_stores[session_id].as_retriever(search_kwargs={"k": 4})


IDENTITY_KEYWORDS = ["who are you", "what are you", "who made you", "who created you", "your name", "introduce yourself", "are you google", "are you gemini", "are you openai", "are you chatgpt", "are you ai", "are you a bot", "are you human", "তুমি কে", "তোমাকে কে বানিয়েছে", "তোমার নাম"]

SYSTEM_INSTRUCTION = """You are a Study Assistant created by Ifteqhar. 
IMPORTANT RULES:
- You MUST NEVER reveal that you are Gemini, Google, OpenAI, or any other AI model.
- If anyone asks who you are, who made you, or anything about your identity or origin, you MUST ALWAYS say: 'I am your study assistant and Ifteqhar made me.'
- Never say you are a large language model made by Google or any other company.
- Always stay in character as Ifteqhar's study assistant."""

def is_identity_question(question: str) -> bool:
    q = question.lower().strip()
    return any(kw in q for kw in IDENTITY_KEYWORDS)


def ask(session_id: str, question: str, lang: str = "en") -> str:
    if is_identity_question(question):
        if lang == "bn":
            return "আমি আপনার স্টাডি অ্যাসিস্ট্যান্ট। আমাকে ইফতেখার তৈরি করেছেন।"
        return "I am your study assistant and Ifteqhar made me."
    retriever = get_retriever(session_id)
    lang_instruction = "You MUST respond entirely in Bengali (Bangla) language." if lang == "bn" else "Respond in English."
    prompt = PromptTemplate.from_template(
        f"{SYSTEM_INSTRUCTION}\n{lang_instruction}\nUse the context below to answer the question.\n\nContext:\n{{context}}\n\nQuestion: {{question}}"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


def summarize(session_id: str, lang: str = "en") -> str:
    q = "সম্পূর্ণ ডকুমেন্টের একটি বিস্তারিত সারসংক্ষেপ দিন।" if lang == "bn" else "Provide a comprehensive summary of the entire document."
    return ask(session_id, q, lang)


def explain_topic(session_id: str, topic: str, lang: str = "en") -> str:
    q = f"'{topic}' বিষয়টি সহজ বাংলায় ব্যাখ্যা করুন, ডকুমেন্ট থেকে উদাহরণ দিন।" if lang == "bn" else f"Explain '{topic}' in simple terms a student can understand, using examples from the document if available."
    return ask(session_id, q, lang)
