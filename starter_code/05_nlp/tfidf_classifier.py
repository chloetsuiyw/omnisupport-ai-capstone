"""Classical NLP starter.
TODO: clean text, compare bag-of-words/TF-IDF, build a ticket classifier, and evaluate errors.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

SUBSET_PATH = Path(__file__).resolve().parents[2] / "data" / "subsets" / "transformer_finetune_10000.parquet"

def load_text_data():
    df = pd.read_parquet(SUBSET_PATH)
    return df

def compare_bow_tfidf(X_train_text, y_train, X_val_text, y_val):
    results = {}
    for name, vectorizer in [("BoW", CountVectorizer(max_features=5000)),
                               ("TF-IDF", TfidfVectorizer(max_features=5000))]:
        X_train_vec = vectorizer.fit_transform(X_train_text)
        X_val_vec = vectorizer.transform(X_val_text)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_val_vec)
        f1 = f1_score(y_val, preds, average="macro")
        results[name] = f1
        print(f"{name}: macro F1 = {f1:.4f}")
    return results

def check_text_diversity(df):
    unique_ratio = df["issue_description"].nunique() / len(df)
    print(f"Unique descriptions: {df['issue_description'].nunique()} / {len(df)} ({unique_ratio:.2%})")
    print("\nSample descriptions from same category:")
    sample_cat = df["issue_category"].iloc[0]
    print(df[df["issue_category"] == sample_cat]["issue_description"].head(5).tolist())

def check_blank_text_errors(X_val_text, y_val, preds):
    X_val_reset = X_val_text.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    is_blank = X_val_reset.astype(str).str.strip() == ""
    print(f"Blank descriptions in validation set: {is_blank.sum()}")
    print(f"Of which misclassified: {((preds != y_val_reset.values) & is_blank.values).sum()}")
    print(f"Total misclassified: {(preds != y_val_reset.values).sum()}")

def train_text_classifier(X_train_text, y_train, X_val_text, y_val):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec = vectorizer.transform(X_val_text)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_val_vec)

    print("\nClassification Report:")
    print(classification_report(y_val, preds))

    return clf, vectorizer, preds

if __name__ == "__main__":
    df = load_text_data()
    print(df.shape)
    print(df["issue_category"].value_counts())

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["issue_description"], df["issue_category"], test_size=0.2, random_state=42, stratify=df["issue_category"]
    )

    print("\n--- Comparing BoW vs TF-IDF ---")
    results = compare_bow_tfidf(X_train_text, y_train, X_val_text, y_val)

    print("\n--- Text Diversity Check ---")
    check_text_diversity(df)

    print("\n--- Final TF-IDF Classifier ---")
    clf, vectorizer, preds = train_text_classifier(X_train_text, y_train, X_val_text, y_val)

    print("\n--- Misclassified Examples ---")
    X_val_reset = X_val_text.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    misclassified = X_val_reset[preds != y_val_reset.values]
    for idx in misclassified.index[:5]:
        print(f"True: {y_val_reset[idx]}, Predicted: {preds[idx]}")
        print(f"Text: {X_val_reset[idx]}\n")

        print("\n--- Blank Text Error Check ---")
    check_blank_text_errors(X_val_text, y_val, preds)