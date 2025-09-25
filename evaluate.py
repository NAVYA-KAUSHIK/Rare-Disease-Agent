# evaluate.py
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

# You need to set your OpenAI API key for the RAGAS judge
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

import config

def load_qa_pipeline():
    """Loads the QA pipeline for evaluation."""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': config.DEVICE}
    )
    db = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs=config.RETRIEVER_SEARCH_KWARGS)
    
    # NOTE: Using the local LLM for generation
    # For RAGAS evaluation, a powerful judge LLM (like GPT-4) is recommended for scoring.
    # The generator LLM can still be your local Mistral model.
    llm = CTransformers(
        model=config.LLM_MODEL_FILE,
        model_type=config.LLM_MODEL_TYPE,
        config={'max_new_tokens': 512, 'temperature': 0.1} # Lower temp for more deterministic eval
    )

    # Simplified prompt for evaluation (non-JSON)
    prompt_template = "CONTEXT: {context}\n\nQUESTION: {question}\n\nINSTRUCTION: Based on the context, provide a direct and concise answer."
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def run_evaluation():
    print("Loading the QA pipeline for evaluation...")
    qa_chain = load_qa_pipeline()

    # --- Create Your "Golden" Test Set ---
    # This is a small, high-quality dataset you create manually.
    # 'ground_truth' is the ideal answer you expect.
    questions = [
        "What are the primary symptoms of Fabry disease?",
        "Which gene is associated with Huntington's disease?",
        "What is the standard treatment for Wilson's disease?",
    ]
    ground_truths = [
        "Primary symptoms of Fabry disease include episodes of pain, particularly in the hands and feet (acroparesthesias), clusters of small, dark red spots on the skin called angiokeratomas, a decreased ability to sweat (hypohidrosis), corneal opacity, and potential kidney and heart problems.",
        "Huntington's disease is associated with a mutation in the HTT gene.",
        "The standard treatment for Wilson's disease involves using chelating agents like penicillamine or trientine to remove excess copper from the body, along with zinc supplements to prevent further copper absorption.",
    ]
    
    print("Running the pipeline on the test set...")
    results = []
    for q in questions:
        result = qa_chain.invoke({"query": q})
        results.append(result)

    # --- Prepare data for RAGAS ---
    answers = [res['result'] for res in results]
    contexts = [[doc.page_content for doc in res['source_documents']] for res in results]

    dataset_dict = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths,
    }
    dataset = Dataset.from_dict(dataset_dict)

    print("Evaluating the pipeline with RAGAS...")
    # NOTE: `ChatOpenAI(model="gpt-4-turbo")` is recommended for the judge
    # If you don't have an OpenAI key, RAGAS may have limited or less reliable functionality.
    ragas_metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]
    
    # Configure the judge LLM
    # A powerful model is needed for reliable scores
    ragas_llm = ChatOpenAI(model_name="gpt-4-turbo")

    result = evaluate(
        dataset=dataset,
        metrics=ragas_metrics,
        llm=ragas_llm, 
    )

    print("\n--- RAGAS Evaluation Results ---")
    print(result)
    
    # Convert to a pandas DataFrame for better viewing
    df = result.to_pandas()
    print("\n--- Results DataFrame ---")
    print(df.to_string())

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set.")
        print("RAGAS evaluation works best with a powerful judge LLM like GPT-4.")
    run_evaluation()