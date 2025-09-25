# build_knowledge_base.py
import os
import json
import time
import torch
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

import config

def create_knowledge_base():
    """
    Processes raw data, creates text files with associated metadata,
    and builds a FAISS vector database.
    """
    print("--- Step 1: Processing Raw Data Files & Creating Metadata ---")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    metadata = {}
    
    # Process PMC Patients
    try:
        print(f"Processing PMC Patients from: {config.PMC_PATIENTS_FILE}")
        with open(config.PMC_PATIENTS_FILE, 'r', encoding='utf-8') as f:
            all_patients_data = json.load(f)
        
        count = 0
        for patient_data in all_patients_data:
            if count >= config.PMC_PATIENTS_LIMIT:
                break
            content = patient_data.get('patient', '')
            if content:
                patient_id = patient_data.get('pmid', f'patient_{count}')
                filename = f"casereport_{patient_id}.txt"
                output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(content)
                metadata[filename] = {"title": f"Case Report: {patient_id}"}
                count += 1
        print(f"Saved {count} case reports from PMC-Patients.")
    except Exception as e:
        print(f"!! ERROR processing PMC Patients: {e}")

    # Process MS^2 Research Papers
    try:
        print(f"Processing MS^2 Research Papers from: {config.MS2_FOLDER}")
        count = 0
        for jsonl_file in os.listdir(config.MS2_FOLDER):
            if not jsonl_file.endswith(".jsonl"):
                continue
            filepath = os.path.join(config.MS2_FOLDER, jsonl_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if count >= config.MS2_PAPERS_LIMIT:
                        break
                    try:
                        data = json.loads(line)
                        text_content = data.get('abstract', '') or data.get('full_text', '')
                        if text_content:
                            doc_id = data.get('id', f'ms2_{count}')
                            title = data.get('title', f"Research Paper {doc_id}")
                            filename = f"research_{doc_id}.txt"
                            output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
                            with open(output_path, 'w', encoding='utf-8') as out_file:
                                out_file.write(text_content)
                            metadata[filename] = {"title": title}
                            count += 1
                    except json.JSONDecodeError:
                        continue
            if count >= config.MS2_PAPERS_LIMIT:
                break
        print(f"Saved {count} research abstracts from MS^2.")
    except Exception as e:
        print(f"!! ERROR processing MS^2 data: {e}")

    # Save metadata
    with open(config.METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {config.METADATA_FILE}")

    print("\n--- Step 2: Creating Vector Database ---")
    os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)

    loader = DirectoryLoader(
        config.PROCESSED_DATA_DIR, glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}, show_progress=True, use_multithreading=True
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} on device: {config.DEVICE}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': config.DEVICE}
    )

    print("Creating FAISS vector database...")
    start_time = time.time()
    db = FAISS.from_documents(texts, embeddings)
    end_time = time.time()
    print(f"Vector database created in {end_time - start_time:.2f} seconds.")

    db.save_local(config.FAISS_INDEX_PATH)
    print(f"Vector database saved successfully to {config.FAISS_INDEX_PATH}!")

if __name__ == "__main__":
    create_knowledge_base()
