from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv
model_name = "AmanSengar/AI-Text-Similarity-Model"
model = SentenceTransformer(model_name)

doc=[
    "A person is having a meal.",
    "The sky is blue today.",
    "Machine learning is fascinating.",
    "A man is eating food."
]

query="A person is eating."

doc_embeddings = model.encode(doc, convert_to_tensor=True)
query_embeddings=model.encode(query,convert_to_tensor=True)

comparisons=util.semantic_search(query_embeddings, doc_embeddings, top_k=2)

print("Query:",query)
print(str(comparisons))
for comparison in comparisons[0]:
    score = comparison['score']
    idx = comparison['corpus_id']
    print(f"Found: '{doc[idx]}' (Score: {score:.4f})")