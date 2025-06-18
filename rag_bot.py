from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from dotenv import load_dotenv
import os
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-..."

app = FastAPI()

index = faiss.read_index("faiss_index.idx")
with open("chunk_data.json") as f:
    data = json.load(f)
chunks = data["chunks"]
sources = data["sources"] 

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def answer_from_docs(query: str) -> str:
    query_vec = embedder.encode([query])
    _, indices = index.search(np.array(query_vec), 3)
    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""You are a helpful assistant. Try to answer the user's question based on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{query}"""

    chat = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
    response = chat([
        SystemMessage(content="Answer the user's question based on the provided context."),
        HumanMessage(content=prompt)
    ])

    return response.content.strip()

rag_tool = Tool(
    name="DocumentQA",
    func=answer_from_docs,
    description="Use this to answer any question about the company's leave policy or internal documents. It retrieves the most relevant context from the documents and answers based on that."
)

llm = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(input: ChatInput):
    try:
        response = agent.run({
            "input": input.message,
            "chat_history": [] 
        })
        return {"response": response}
    except Exception as e:
        return {"response": f"Agent Error: {str(e)}"}
