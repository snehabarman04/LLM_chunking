import re
import fitz  # PyMuPDF
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    
    doc = fitz.open(pdf_path)
    content = ""
    for page in doc:
        content += page.get_text()
    return content


def split_into_units(text):
    text = re.sub(r"\n\s*\n", "\n\n", text)
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def generate_embeddings(paragraphs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(paragraphs)
    return embeddings


def cluster_paragraphs(paragraphs, embeddings, distance_threshold=0.8):
    similarity_matrix = cosine_similarity(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage='average',
        distance_threshold=1 - distance_threshold
    )
    
    distance_matrix = 1 - similarity_matrix
    cluster_labels = clustering.fit_predict(distance_matrix)

    clustered_paragraphs = {}
    for idx, label in enumerate(cluster_labels):
        clustered_paragraphs.setdefault(label, []).append(paragraphs[idx])

    chunks = ["\n".join(clustered_paragraphs[label]) for label in sorted(clustered_paragraphs.keys())]
    return chunks

def semantic_chunking(pdf_path, distance_threshold=0.8):
    
    text = extract_text_from_pdf(pdf_path)
    units = split_into_units(text)
    embeddings = generate_embeddings(units)
    chunks = cluster_paragraphs(units, embeddings, distance_threshold=distance_threshold)

    return chunks


if __name__ == "__main__":
    pdf_path = r"C:\Users\SNEHA\Downloads\CompanyLease Vehicle FAQ.pdf"
    chunks = semantic_chunking(pdf_path, distance_threshold=0.8)

    for idx, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---")
        print(chunk)
        print("\n" + "-" * 50 + "\n")
