"""Session 20-22 transformer starter.

Mandatory student work:
1. Demonstrate query/key/value, self-attention and positional information using
   one support message; include a worked example or visualisation and contrast
   it with TF-IDF.
2. Use pretrained transformer pipelines for at least two appropriate tasks.
3. Fine-tune/adapt a small Hugging Face text classifier on the supplied
   10,000-row subset and compare it with the classical TF-IDF baseline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

SUBSET_PATH = Path(__file__).resolve().parents[2] / "data" / "subsets" / "transformer_finetune_10000.parquet"


def load_finetuning_subset():
    return pd.read_parquet(SUBSET_PATH)


def prepare_training_subset(df, n=10000):
    """Deterministic helper only; model/tokenizer/training/evaluation remain student tasks."""
    return df.sample(min(n, len(df)), random_state=42).copy()


def demonstrate_attention(message: str):
    """Manual self-attention worked example: tokenize, embed, add positional
    encoding, compute Q/K/V, and show attention weights between tokens."""
    torch.manual_seed(42)

    tokens = message.lower().replace(".", "").replace(",", "").split()
    vocab = {word: i for i, word in enumerate(sorted(set(tokens)))}
    token_ids = torch.tensor([vocab[t] for t in tokens])

    d_model = 8
    n_tokens = len(tokens)

    embedding = torch.nn.Embedding(len(vocab), d_model)
    x = embedding(token_ids)  # (n_tokens, d_model)

    # Positional encoding (sinusoidal, standard transformer approach)
    position = torch.arange(n_tokens).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pos_encoding = torch.zeros(n_tokens, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    x_with_pos = x + pos_encoding

    # Q, K, V projections
    W_q = torch.nn.Linear(d_model, d_model, bias=False)
    W_k = torch.nn.Linear(d_model, d_model, bias=False)
    W_v = torch.nn.Linear(d_model, d_model, bias=False)

    Q = W_q(x_with_pos)
    K = W_k(x_with_pos)
    V = W_v(x_with_pos)

    # Scaled dot-product self-attention
    scores = Q @ K.T / np.sqrt(d_model)
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V

    attention_df = pd.DataFrame(attention_weights.detach().numpy(), index=tokens, columns=tokens)

    return {
        "tokens": tokens,
        "attention_weights": attention_df,
        "output": output,
    }


def plot_attention_heatmap(attention_df, save_path="outputs/06_transformers/attention_heatmap.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attention_df.values, cmap="viridis")
    ax.set_xticks(range(len(attention_df.columns)))
    ax.set_yticks(range(len(attention_df.index)))
    ax.set_xticklabels(attention_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(attention_df.index)
    ax.set_xlabel("Key (attended-to token)")
    ax.set_ylabel("Query (attending token)")
    ax.set_title("Self-Attention Weights")
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def contrast_with_tfidf(message, attention_df):
    """Show TF-IDF's bag-of-words vector (no order, no relationships) vs
    attention's token-to-token relationship matrix, for direct contrast."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_vec = vectorizer.fit_transform([message])
    tfidf_series = pd.Series(tfidf_vec.toarray()[0], index=vectorizer.get_feature_names_out())
    return tfidf_series.sort_values(ascending=False)


def run_pretrained_pipelines(sample_texts):
    from transformers import pipeline

    print("Loading sentiment-analysis pipeline...")
    sentiment_pipe = pipeline("sentiment-analysis")
    sentiment_results = sentiment_pipe(sample_texts)

    print("Loading zero-shot-classification pipeline...")
    zero_shot_pipe = pipeline("zero-shot-classification")
    candidate_labels = ["damaged item", "delivery problem", "refund request", "account issue", "general question"]
    zero_shot_results = [zero_shot_pipe(text, candidate_labels) for text in sample_texts]

    return sentiment_results, zero_shot_results


def fine_tune_text_classifier(train_df, validation_df, sample_size=1500, epochs=2):
    """Fine-tune a small DistilBERT classifier on a reduced sample, for CPU tractability.
    This sample size and epoch count are deliberate compute-time decisions, documented
    explicitly given the lack of GPU access."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from torch.utils.data import Dataset as TorchDataset

    train_sample = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    val_sample = validation_df.sample(min(300, len(validation_df)), random_state=42)

    labels = sorted(train_sample["issue_category"].unique())
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(labels)
    )

    class TicketTextDataset(TorchDataset):
        def __init__(self, df):
            self.texts = df["issue_description"].tolist()
            self.labels = [label_to_id[l] for l in df["issue_category"]]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = TicketTextDataset(train_sample)
    val_ds = TicketTextDataset(val_sample)

    training_args = TrainingArguments(
        output_dir="outputs/06_transformers/finetune_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    eval_results = trainer.evaluate()

    preds_output = trainer.predict(val_ds)
    preds = preds_output.predictions.argmax(axis=1)
    true_labels = [label_to_id[l] for l in val_sample["issue_category"]]

    return trainer, model, tokenizer, eval_results, preds, true_labels, id_to_label


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    sample_message = "delivery driver dropped my parcel outside despite clear instructions"
    result = demonstrate_attention(sample_message)

    print("Tokens:", result["tokens"])
    print("\nAttention weight matrix (rows = query token, columns = key token):")
    print(result["attention_weights"].round(3))

    plot_attention_heatmap(result["attention_weights"])

    print("\n--- TF-IDF representation of the same message (for contrast) ---")
    tfidf_result = contrast_with_tfidf(sample_message, result["attention_weights"])
    print(tfidf_result)

    print("\n--- Pretrained Transformer Pipelines ---")
    df = load_finetuning_subset()
    sample_texts = df[df["issue_description"].astype(str).str.strip() != ""]["issue_description"].head(3).tolist()

    sentiment_results, zero_shot_results = run_pretrained_pipelines(sample_texts)

    for i, text in enumerate(sample_texts):
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment_results[i]}")
        print(f"Zero-shot top label: {zero_shot_results[i]['labels'][0]} ({zero_shot_results[i]['scores'][0]:.3f})")

    print("\n--- Fine-Tuning DistilBERT (Session 22, Mandatory) ---")
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import f1_score as f1s

    full_df = load_finetuning_subset()
    full_df = full_df[full_df["issue_description"].astype(str).str.strip() != ""]
    train_ft_df, val_ft_df = tts(full_df, test_size=0.2, random_state=42, stratify=full_df["issue_category"])

    trainer, ft_model, ft_tokenizer, eval_results, preds, true_labels, id_to_label = fine_tune_text_classifier(
        train_ft_df, val_ft_df, sample_size=1500, epochs=2
    )
    print("\nEval results:", eval_results)

    ft_f1 = f1s(true_labels, preds, average="macro")
    print(f"\nFine-tuned model macro F1: {ft_f1:.4f}")
    print("(Compare to TF-IDF baseline macro F1: 0.9925 from Phase 8a)")