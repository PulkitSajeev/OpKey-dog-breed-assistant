import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_DATA_PATH = "src/data/processed_data.csv"
EMBEDDINGS_PATH = "src/data/embeddings.pkl"

def generate_embeddings():
    # Generates Sentence-BERT embeddings for dog breed descriptions.
    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    embeddings = {row["breed"]: model.encode(row["description"]) for _, row in df.iterrows()}

    # Save embeddings
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved at: {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    generate_embeddings()
