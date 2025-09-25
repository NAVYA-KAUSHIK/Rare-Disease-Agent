# config.py
import os
import torch

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
LLM_MODEL_DIR = os.path.join(BASE_DIR, "llm_model")

# --- Data Source Paths ---
MS2_FOLDER = os.path.join(RAW_DATA_DIR, "ms2_data")
PMC_PATIENTS_FILE = os.path.join(RAW_DATA_DIR, "PMC-Patients.json")
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")

# --- Vector DB ---
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index")

# --- Model Configuration ---
# Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LLM
LLM_MODEL_FILE = os.path.join(LLM_MODEL_DIR, "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
LLM_MODEL_TYPE = "mistral"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
CONTEXT_LENGTH = 4096

# --- Retriever Configuration ---
RETRIEVER_SEARCH_KWARGS = {'k': 3}

# --- Data Processing Limits ---
PMC_PATIENTS_LIMIT = 2000
MS2_PAPERS_LIMIT = 2000