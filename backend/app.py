# app.py
import os
import json
import re
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

import config

class PatientQuery(BaseModel):
    symptoms: str
    age: int
    gender: str
    genetic_test: str | None = None

app = FastAPI(title="Rare Disease Diagnosis API", description="An AI-powered Medical Research Agent")
ml_models = {}
metadata = {}

@app.on_event("startup")
async def startup_event():
    print("--> Loading models, vector DB, and metadata...")
    
    global metadata
    if os.path.exists(config.METADATA_FILE):
        with open(config.METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print("Metadata loaded.")

    # Load embedding model and vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': config.DEVICE}
    )
    db = FAISS.load_local(
        config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs=config.RETRIEVER_SEARCH_KWARGS)

    # Load LLM
    llm = CTransformers(
        model=config.LLM_MODEL_FILE,
        model_type=config.LLM_MODEL_TYPE,
        config={
            'max_new_tokens': config.MAX_NEW_TOKENS,
            'temperature': config.TEMPERATURE,
            'context_length': config.CONTEXT_LENGTH
        }
    )

    prompt_template = """
    INSTRUCTION: You are an AI medical expert. Analyze the CONTEXT to answer the QUESTION.
    Your answer must be ONLY a JSON object with keys "possible_diagnosis", "suggested_treatment", and "confidence_score".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    JSON_RESPONSE:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": prompt}
    )
    
    ml_models["qa_chain"] = qa_chain
    print("--> Models and vector DB loaded successfully!")

@app.post("/diagnose")
async def diagnose_patient(query: PatientQuery):
    if "qa_chain" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    
    patient_summary = (
        f"Diagnose the condition for a {query.age}-year-old {query.gender} with symptoms: {query.symptoms}."
        + (f" Genetic test results show: {query.genetic_test}." if query.genetic_test else "")
    )
    
    try:
        result = await asyncio.to_thread(
            ml_models["qa_chain"].invoke, {"query": patient_summary}
        )
        
        sources = []
        for doc in result["source_documents"]:
            filename = os.path.basename(doc.metadata.get('source', 'N/A'))
            meta = metadata.get(filename, {})
            sources.append({"filename": filename, "title": meta.get("title", "N/A")})

        try:
            json_match = re.search(r'\{.*\}', result["result"], re.DOTALL)
            if json_match:
                llm_response_json = json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON object found in the LLM response.")
            
            response_data = {
                "possible_diagnosis": llm_response_json.get("possible_diagnosis", "Diagnosis not available"),
                "suggested_treatment": llm_response_json.get("suggested_treatment", "Treatment plan not available"),
                "confidence_score": llm_response_json.get("confidence_score", 0.0),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error parsing LLM JSON response: {e}")
            response_data = {"raw_text_response": result["result"]}

        return {
            "patient_summary": patient_summary,
            "structured_response": response_data,
            "retrieved_sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during diagnosis: {str(e)}")
