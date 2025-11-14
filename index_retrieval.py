from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-mpnet-base-v2')
passages = [...]  # list of texts
embeddings = encoder.encode(passages, convert_to_tensor=True)
