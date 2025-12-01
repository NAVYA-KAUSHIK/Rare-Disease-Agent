# model/app.py
import os
import json
import re
import asyncio
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Configuration Setup ---
# (Duplicating config here to ensure app runs standalone without import errors)
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root, then into vector_db
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "..", "vector_db")
    FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
    METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
    
    # Model Paths - Adjust these filenames if yours are different!
    MODEL_DIR = os.path.join(BASE_DIR, "..", "llm_model")
    LLM_MODEL_FILE = os.path.join(MODEL_DIR, "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    EMBEDDING_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    
    DEVICE = "cpu"
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.01  # Ultra-low temperature for factual accuracy
    CONTEXT_LENGTH = 4096

config = Config()

# --- Data Models ---
class PatientQuery(BaseModel):
    symptoms: str
    age: int | str
    gender: str
    genetic_test: str | None = None

app = FastAPI(
    title="Rare Disease Copilot",
    description="AI-powered diagnostic assistant using RAG with Mistral-7B"
)

# Global variables to hold models in memory
ml_models = {}
metadata_store = {}

@app.on_event("startup")
async def startup_event():
    print("--- 🚀 Starting Rare Disease Copilot ---")
    
    # 1. Load Metadata
    global metadata_store
    if os.path.exists(config.METADATA_FILE):
        with open(config.METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata_store = json.load(f)
        print(f"✅ Loaded metadata for {len(metadata_store)} documents.")
    else:
        print("⚠️ Warning: Metadata file not found.")

    # 2. Load Vector DB
    print(f"Loading Vector DB from: {config.FAISS_INDEX_PATH}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME, 
            model_kwargs={'device': config.DEVICE}
        )
        db = FAISS.load_local(
            config.FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        # Search kwargs: fetch more docs (k=5) to ensure we get relevant hits
        retriever = db.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector Database loaded.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading Vector DB: {e}")
        raise e

    # 3. Load Mistral LLM
    print(f"Loading Mistral Model from: {config.LLM_MODEL_FILE}")
    try:
        llm = CTransformers(
            model=config.LLM_MODEL_FILE,
            model_type="mistral",
            config={
                'max_new_tokens': config.MAX_NEW_TOKENS,
                'temperature': config.TEMPERATURE,
                'context_length': config.CONTEXT_LENGTH,
                'gpu_layers': 0  # Set to > 0 if you move to a GPU instance
            }
        )
        print("✅ Mistral LLM loaded.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading LLM: {e}")
        raise e

    # 4. Prompt Engineering (The "Geneticist Persona")
    # This specific format [INST] ... [/INST] is required for Mistral to follow instructions
    template = """[INST] You are an expert Senior Clinical Geneticist specializing in rare diseases. 
    Use the following pieces of retrieved medical context to diagnose the patient.
    
    RULES:
    1. Answer ONLY based on the context provided. If the context does not contain the answer, say "Insufficient clinical data."
    2. Provide a structured JSON response. Do not add any conversational text outside the JSON.
    3. The JSON must have these keys: "diagnosis", "reasoning", "treatment_plan", "confidence_score" (0-100).

    CONTEXT:
    {context}

    PATIENT CASE:
    {question}

    Generate JSON response: [/INST]
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    ml_models["qa_chain"] = qa_chain
    print("--- System Ready ---")

@app.post("/diagnose")
async def diagnose_patient(query: PatientQuery):
    if "qa_chain" not in ml_models:
        raise HTTPException(status_code=503, detail="System initializing...")

    # Construct a clinical summary string
    patient_summary = (
        f"Patient Demographics: {query.age} year old {query.gender}. "
        f"Clinical Presentation: {query.symptoms}. "
    )
    if query.genetic_test:
        patient_summary += f"Genetic Findings: {query.genetic_test}."

    print(f"\n🔎 Analyzing: {patient_summary}")
    start_time = time.time()

    try:
        # Run inference in a thread to prevent blocking the API
        result = await asyncio.to_thread(
            ml_models["qa_chain"].invoke, {"query": patient_summary}
        )
        
        # --- Post-Processing ---
        raw_response = result["result"]
        source_docs = result["source_documents"]
        
        # 1. Extract Sources
        sources = []
        for doc in source_docs:
            # Try to match the source file to our metadata
            source_path = doc.metadata.get('source', '')
            filename = os.path.basename(source_path)
            meta = metadata_store.get(filename, {})
            
            sources.append({
                "source": meta.get("title", filename),
                "type": meta.get("type", "Unknown"),
                "snippet": doc.page_content[:100] + "..." # Preview
            })

        # 2. Parse JSON safely
        # Mistral sometimes puts text before/after the JSON, so we use regex to find the { } block
        try:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group(0))
            else:
                # Fallback if model refuses to output JSON
                structured_data = {
                    "diagnosis": "Parsing Error",
                    "reasoning": raw_response,
                    "treatment_plan": "N/A",
                    "confidence_score": 0
                }
        except Exception:
            structured_data = {
                "diagnosis": "Format Error",
                "reasoning": raw_response, 
                "confidence_score": 0
            }

        return {
            "status": "success",
            "inference_time": f"{time.time() - start_time:.2f}s",
            "analysis": structured_data,
            "evidence_sources": sources
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)