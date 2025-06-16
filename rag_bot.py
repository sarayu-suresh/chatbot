from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import torch
import fitz  

app = FastAPI()

model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    all_text = []
    for page in doc:
        text = page.get_text()
        all_text.append(text)
    return "\n".join(all_text)

text = extract_text_from_pdf("data/leave_policy.pdf")
chunks = text.split("\n\n")  

chunk_embeddings = embedder.encode(chunks)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(input: ChatInput):
    query = input.message
    query_vec = embedder.encode([query])
    top_k = 1
    _, indices = index.search(query_vec, top_k)
    
    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"Instruction: Try to understand and answer by yourself based on the info in the context.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to("cpu")
    attention_mask = inputs.attention_mask.to("cpu")

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,  
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )


    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    parts = raw_output.split("Answer:")
    if len(parts) > 1:
        response = parts[1].strip().split("Question:")[0].strip()
    else:
        response = raw_output.strip()



    return {"response": response}

