import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

KB_folder='knowledgebase'


category_indices = {}
all_docs = []
all_embeddings = None  
faiss_index = None     

def load_kb():
    kb={}
    for filename in os.listdir(KB_folder):
        if filename.endswith(".txt"):
            category=filename.replace(".txt","").capitalize()
            with open(os.path.join(KB_folder, filename), "r", encoding="utf-8") as f:
                docs=[line.strip() for line in f.readlines() if line.strip()]
            kb[category]=docs
    return kb
        
def populate_db():
    print(" Populating FAISS DB with embeddings...")
    

    global all_embeddings, all_docs, category_indices, faiss_index
    all_docs=[]
    category_indices={}
    all_embeddings=None
    faiss_index=None
    
    all_embeddings_list = []
    idx_offset = 0
    knowledge_base=load_kb()

    for category, docs in knowledge_base.items():
        embeddings = embed_model.encode(docs)
        all_embeddings_list.append(embeddings)
        all_docs.extend(docs)
        category_indices[category] = (idx_offset, idx_offset + len(docs))
        idx_offset += len(docs)

    # Convert all embeddings to a single numpy array
    all_embeddings = np.vstack(all_embeddings_list).astype("float32")

    # Initialize FAISS index
    dim = all_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(all_embeddings)

    print(f" Added {len(all_docs)} documents to FAISS index.")
def retrieve_context(
    category: str,
    subject: str,
    description: str,
    top_k: int = 3,
    attempt: int = 1,
    classification_scores: list = None,
    feedback: str = None,   
):
    """
    Retrieve context passages given a category + query.
    On retry, feedback is appended to the query to bias retrieval.
    """

    query_text = f"{subject} {description}"

    if feedback:
        query_text += f" | Reviewer feedback: {feedback}"

    query_embedding = embed_model.encode([query_text]).astype("float32")

    # get docs for category
    start_idx, end_idx = category_indices.get(category, (0, len(all_docs)))
    category_embeddings = all_embeddings[start_idx:end_idx]
    category_docs = all_docs[start_idx:end_idx]

    temp_index = faiss.IndexFlatL2(category_embeddings.shape[1])
    temp_index.add(category_embeddings)

    distances, indices = temp_index.search(query_embedding, len(category_docs))

    print("retriever attempt:", attempt)
    print("classification scores:", classification_scores)
    print("feedback used:", feedback)

    if attempt == 0 or attempt == 2:  
        # normal — best matches
        selected = indices[0][:top_k]
    else:
        # retry — weaker matches
        selected = indices[0][top_k:top_k + top_k]

    return [category_docs[i] for i in selected]
