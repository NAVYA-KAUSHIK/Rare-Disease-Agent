# model/build_knowledge_base.py
import os
import json
import time
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    import config
except ImportError:
    # Fallback if config.py isn't found locally while testing
    class Config:
        RAW_DATA_PATH = os.path.join("..", "data", "raw")
        PROCESSED_DATA_DIR = os.path.join("..", "data", "processed")
        VECTOR_DB_DIR = os.path.join("..", "vector_db")
        FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index")
        MS2_FOLDER = os.path.join(RAW_DATA_PATH, "ms2_data")
        PMC_PATIENTS_FILE = os.path.join(RAW_DATA_PATH, "PMC-Patients.json")
        METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
        EMBEDDING_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        DEVICE = "cpu"
        PMC_PATIENTS_LIMIT = 2000
        MS2_PAPERS_LIMIT = 2000
    config = Config()

def clean_text(text):
    """Basic cleaning to remove excessive whitespace or artifacts."""
    if not text:
        return ""
    return " ".join(text.split())

def format_patient_record(data, index):
    """
    Transforms raw JSON into a semantically rich text format.
    Uses 'index' to ensure a unique filename if the PMID is missing.
    """
    raw_id = data.get('pmid')
    
    if not raw_id:
        patient_id = f"patient_{index}"
    else:
        patient_id = str(raw_id).replace('/', '-').replace('\\', '-')
    
    age = data.get('age', 'Unknown')
    gender = data.get('gender', 'Unknown')
    narrative = data.get('patient', '')
    
    structured_text = (
        f"--- CLINICAL CASE REPORT ---\n"
        f"SOURCE ID: {patient_id}\n"
        f"PATIENT DEMOGRAPHICS: Age {age}, Gender {gender}\n"
        f"CLINICAL HISTORY & SYMPTOMS:\n{narrative}\n"
    )
    return structured_text, patient_id

def create_knowledge_base():
    print("--- Step 1: Processing Raw Data (Structured formatting) ---")
    
    if not os.path.exists(config.PROCESSED_DATA_DIR):
        os.makedirs(config.PROCESSED_DATA_DIR)
    
    metadata_store = {}
    
    # 1. Process PMC Patients (Clinical Cases)
    try:
        print(f"Processing PMC Patients from: {config.PMC_PATIENTS_FILE}")
        count = 0
        
        with open(config.PMC_PATIENTS_FILE, 'r', encoding='utf-8') as f:
            all_patients = json.load(f)
            
        for patient_data in all_patients:
            if count >= config.PMC_PATIENTS_LIMIT:
                break
                
            structured_content, pmid = format_patient_record(patient_data,count)
            
            if structured_content:
                filename = f"casereport_{pmid}.txt"
                output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
                
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(structured_content)
                
                metadata_store[filename] = {
                    "title": f"Case Report {pmid}",
                    "type": "clinical_case",
                    "source_id": pmid
                }
                count += 1
                
        print(f"✅ Processed {count} PMC patient records.")
        
    except Exception as e:
        print(f"!! ERROR processing PMC Patients: {e}")

    # 2. Process MS^2 (Research Papers)
    try:
        print(f"Processing MS^2 Research Papers from: {config.MS2_FOLDER}")
        count = 0
        if os.path.exists(config.MS2_FOLDER):
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
                            title = data.get('title', 'Unknown Title')
                            abstract = data.get('abstract', '') or data.get('full_text', '') or data.get('text', '')
                            doc_id = data.get('id', f'ms2_{count}')
                            
                            if abstract:
                                # Structured format for papers
                                structured_content = (
                                    f"--- MEDICAL RESEARCH PAPER ---\n"
                                    f"TITLE: {title}\n"
                                    f"ABSTRACT/CONTENT:\n{clean_text(abstract)}\n"
                                )
                                
                                filename = f"research_{doc_id}.txt"
                                output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
                                
                                with open(output_path, 'w', encoding='utf-8') as out_file:
                                    out_file.write(structured_content)
                                    
                                metadata_store[filename] = {
                                    "title": title,
                                    "type": "research_paper",
                                    "source_id": doc_id
                                }
                                count += 1
                        except json.JSONDecodeError:
                            continue
        print(f"✅ Processed {count} MS^2 research abstracts.")
        
    except Exception as e:
        print(f"!! ERROR processing MS^2 data: {e}")

    # Save Metadata
    with open(config.METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_store, f, indent=4)
    print(f"Metadata saved to {config.METADATA_FILE}")

    print("\n--- Step 2: Creating Vector Database (optimized chunking) ---")
    
    loader = DirectoryLoader(
        config.PROCESSED_DATA_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    if not documents:
        print("!! CRITICAL ERROR: No documents found to embed. Check data paths.")
        return

   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")

    print(f"Loading Embedding Model ({config.EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': config.DEVICE},
        encode_kwargs={'normalize_embeddings': True} # Normalization helps cosine similarity
    )

    print("Building FAISS Index...")
    start_time = time.time()
    db = FAISS.from_documents(texts, embeddings)
    print(f"Vector DB built in {time.time() - start_time:.2f} seconds.")

    db.save_local(config.FAISS_INDEX_PATH)
    print(f"✅ Success! Database saved to: {config.FAISS_INDEX_PATH}")

if __name__ == "__main__":
    create_knowledge_base()