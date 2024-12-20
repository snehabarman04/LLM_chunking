import re
import fitz
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams


class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data):
        new_node = Node(data)
        if self.tail:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        else:
            self.head = self.tail = new_node

    def traverse_forward(self):
        current = self.head
        while current:
            print("----------- Node Start -----------")
            print(current.data)
            print("------------ Node End ------------\n")
            current = current.next


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    content = ""
    for page in doc:
        content += page.get_text()
    return content


def split_into_sections(text):
    qa_pattern = re.compile(r"^\s*\d+\.\s+.*\?", re.MULTILINE)
    matches = list(qa_pattern.finditer(text))

    if matches:
        sections = []
        for i, match in enumerate(matches):
            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(text[start_idx:end_idx].strip())
        return sections
    else:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]


def pdf_to_doubly_linked_list(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(text)
    dll = DoublyLinkedList()
    for section in sections:
        dll.append(section)

    return dll

if __name__ == "__main__":
    pdf_path = r"C:/Users/SNEHA/Downloads/CompanyLease Vehicle FAQ.pdf"

    doubly_linked_list = pdf_to_doubly_linked_list(pdf_path)

    print("Traversing forward:")
    doubly_linked_list.traverse_forward()

    print("\nTraversing backward:")
    doubly_linked_list.traverse_backward()


def generate_embeddings_for_nodes_sequentially(dll):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    current = dll.head
    idx = 0
    while current:
        embedding = model.encode(current.data)
        embeddings.append((idx, embedding, current.data)) 
        current = current.next
        idx += 1
    return embeddings


def index_to_qdrant_sequential(embeddings):
    client = QdrantClient(host="localhost", port=6333)  

    vector_size = len(embeddings[0][1])  
    client.recreate_collection(
        collection_name="faq_sections",
        vectors_config=VectorParams(
            size=vector_size,  
            distance="Cosine"  
        )
    )

    points = [
        PointStruct(
            id=idx,
            vector=embedding,
            payload={"text": text_data, "index": idx}
        )
        for idx, embedding, text_data in embeddings
    ]
    client.upsert(collection_name="faq_sections", points=points)

def query_qdrant(query, model):
    client = QdrantClient(host="localhost", port=6333) 
    query_vector = model.encode([query])[0]

    search_results = client.search(
        collection_name="faq_sections",
        query_vector=query_vector,
        limit=5 
    )

    print("\nResults with context:")
    for result in search_results:
        print("Match Score:", result.score)
        print("Text:", result.payload["text"])
        print("Index:", result.payload["index"])
        print("--------------------")


if __name__ == "__main__":
    pdf_path = r"C:\Users\SNEHA\Downloads\CompanyLease Vehicle FAQ.pdf"

    doubly_linked_list = pdf_to_doubly_linked_list(pdf_path)

    embeddings = generate_embeddings_for_nodes_sequentially(doubly_linked_list)
    index_to_qdrant_sequential(embeddings)
    print("\nIndexed sections into Qdrant. You can now query.")


    user_query = input("\nEnter your query: ")    
    print("\nQuerying Qdrant:")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_qdrant(user_query, model)
