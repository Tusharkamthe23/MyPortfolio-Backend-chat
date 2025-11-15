import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableMap, RunnablePassthrough

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

FAISS_PATH = "faiss_mistral_v1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Embeddings ---
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)

# --- Load FAISS ---
db = FAISS.load_local(
    FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# --- LLM ---
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# --- Format retrieved docs ---
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# --- RAG pipeline (LangChain 1.0.7 compliant) ---
rag_chain = (
    RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | (lambda x: f"Context:\n{x['context']}\n\nQuestion: {x['question']}\n\nAnswer:")
    | llm
)

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    user_input = query.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        answer = rag_chain.invoke(user_input)
        return {"response": answer.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
