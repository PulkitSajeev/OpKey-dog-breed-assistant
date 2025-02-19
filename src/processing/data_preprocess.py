import pandas as pd
import re

DATA_PATH = "src/data/dog-breed.csv"
PROCESSED_DATA_PATH = "src/data/processed_data.csv"

def clean_text(text):
    # Removes special characters and converts text to lowercase.
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(text)).lower()

def preprocess_data():
    # Loads, cleans, and processes dataset.
    df = pd.read_csv(DATA_PATH)

    # Rename breed column
    df.rename(columns={"Unnamed: 0": "breed"}, inplace=True)

    # Clean text columns
    df["temperament"] = df["temperament"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved at: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()
