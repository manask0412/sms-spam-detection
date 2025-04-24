import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text  # noqa: loads the USE ops
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------ Config -------------------
TEST_CSV    = 'data/evaluation.csv'
MODEL_PATH  = 'models/svm_spam_classifier.joblib'
USE_URL     = "https://tfhub.dev/google/universal-sentence-encoder/4"

# -------------- Data Loading -----------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding='latin-1')
    if set(['v1','v2']).issubset(df.columns):
        df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})
    elif set(['label','text']).issubset(df.columns):
        df = df[['label','text']]
    else:
        raise ValueError("CSV must have columns ['v1','v2'] or ['label','text']")
    df = df.dropna(subset=['label','text'])
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'ham':0,'spam':1})
    return df['text'].tolist(), df['label'].astype(int).values

# ------------------ Embedding Helper ------------------
def embed_texts(texts):
    """Batch-embed a list of strings via USE."""
    use = hub.load(USE_URL)
    return use(texts).numpy()

# ------------------ Evaluate & Plot ------------------

def evaluate_svm_model(pipe, thresh, X_texts, y_true):
    # embed
    X_emb = embed_texts(X_texts)
    
    # predict probabilities
    y_prob = pipe.predict_proba(X_emb)[:,1]

    # use saved validation threshold
    print(f"\nUsing validation‐tuned threshold = {thresh:.2f}")

    # final predictions
    y_pred = (y_prob > thresh).astype(int)

    # metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['ham','spam']))
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"Mismatches: {np.sum(y_true!=y_pred)} / {len(y_true)}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['ham','spam'],
                yticklabels=['ham','spam'],
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print(f"AUC = {roc_auc:.4f}")

print("Loading evaluation data...")
X_test, y_test = load_dataset(TEST_CSV)

print("Loading SVM pipeline + saved threshold...")
data     = joblib.load(MODEL_PATH)
svm_pipe = data['model']
thresh   = data['thresh']  # use the threshold you saved from validation

print("Evaluating on sample messages....")
evaluate_svm_model(svm_pipe, thresh, X_test, y_test)