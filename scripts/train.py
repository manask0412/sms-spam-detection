import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_text  # noqa
import nltk
import joblib

from nltk.corpus import wordnet, stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------- NLTK Setup -----------------
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------ Config -------------------
SEED        = 42
MODEL_URL   = "https://tfhub.dev/google/universal-sentence-encoder/4"
MODEL_PATH  = "models/svm_spam_classifier.joblib"
PARAM_GRID  = {
    'svm__C':      [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma':  ['scale', 'auto']
}
train_csv = 'data/train.csv'
test_csv  = 'data/test.csv'

# -------------- Data Loading -----------------
def load_dataset(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})
    df['label'] = df['label'].map({'ham':0,'spam':1})
    return df

# ---------- Augmentation Helpers ------------
def synonym_replacement(text, replace_prob=0.3):
    words, out = text.split(), []
    for w in words:
        if w.lower() in stop_words or random.random()>replace_prob:
            out.append(w)
        else:
            syns = wordnet.synsets(w)
            if syns and len(syns[0].lemmas())>1:
                out.append(random.choice(syns[0].lemmas()[1:]).name().replace('_',' '))
            else:
                out.append(w)
    return ' '.join(out)

def introduce_typos(text, typo_prob=0.3):
    words = text.split()
    for i in range(len(words)):
        if random.random()<typo_prob and len(words[i])>2:
            c = list(words[i])
            idx = random.randint(1,len(c)-2)
            c[idx],c[idx+1] = c[idx+1],c[idx]
            words[i] = ''.join(c)
    return ' '.join(words)

def augment_dataset(df, syn_frac=0.3, typo_frac=0.3):
    syn_idx  = df.sample(frac=syn_frac, random_state=SEED).index
    typo_idx = df.sample(frac=typo_frac, random_state=SEED+1).index
    aug_rows = []
    for i in syn_idx:
        aug_rows.append({'label':df.at[i,'label'],'text':synonym_replacement(df.at[i,'text'])})
    for i in typo_idx:
        aug_rows.append({'label':df.at[i,'label'],'text':introduce_typos(df.at[i,'text'])})
    return pd.concat([df,pd.DataFrame(aug_rows)]).sample(frac=1, random_state=SEED)

# ---------- Embedding Helper ----------------
def embed_texts(texts):
    use = hub.load(MODEL_URL)
    return use(texts).numpy()

# --------- Plotting Utilities -------------
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['ham','spam'],
                yticklabels=['ham','spam'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

def plot_grid_scores(grid):
    """Heatmap of mean F1 for each C × kernel (averaging over gamma)."""
    results = pd.DataFrame(grid.cv_results_)
    # average mean_test_score over gamma
    pivot = (results
             .groupby(['param_svm__C','param_svm__kernel'])
             ['mean_test_score']
             .mean()
             .unstack())
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
    plt.title('GridSearchCV Mean F1‑Score')
    plt.xlabel('Kernel'); plt.ylabel('C')
    plt.show()

def plot_threshold_metrics(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2*(precision*recall)/(precision+recall+1e-8)
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1],    label='Recall')
    plt.plot(thresholds, f1_scores[:-1],  label='F1 Score')
    plt.axvline(0.5, color='gray', linestyle='--', label='Thresh=0.5')
    plt.title('Threshold Tuning Metrics')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.legend(); plt.grid(True)
    plt.show()

# -------- Grid Search for SVM -------------
def svm_grid_search(X_embed, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(probability=True, class_weight='balanced', random_state=SEED))
    ])
    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring='f1',
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    grid.fit(X_embed, y)
    print("🔍 Best Params:", grid.best_params_)
    print("🏅 Best CV F1 :", grid.best_score_)
    # new plot:
    plot_grid_scores(grid)
    return grid.best_estimator_

# -------------- Main Pipeline ----------------
# 1. Load & augment
df      = load_dataset(train_csv)
df_aug  = augment_dataset(df)

# 2. Split
X = df_aug['text'].tolist()
y = df_aug['label'].values
X_train, X_val, y_train, y_val = train_test_split(
     X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 3. Embed
print("Embedding train & val...")
X_train_emb = embed_texts(X_train)
X_val_emb   = embed_texts(X_val)

# 4. GridSearch + train
svm_model = svm_grid_search(X_train_emb, y_train)

# 5. Threshold tuning on val
val_prob = svm_model.predict_proba(X_val_emb)[:,1]
best_t, best_f1 = 0.5, 0
for t in np.linspace(0.1,0.9,17):
    f = f1_score(y_val, (val_prob>t).astype(int))
    if f>best_f1:
        best_f1, best_t = f, t
print(f"🎯 Best threshold={best_t:.2f}, Val F1={best_f1:.3f}")

# plot threshold metrics
plot_threshold_metrics(y_val, val_prob)

# 6. Save model + thresh
joblib.dump({'model':svm_model,'thresh':best_t}, MODEL_PATH)
print(f"💾 Saved to {MODEL_PATH}")

# 7. Test evaluation
df_test = load_dataset(test_csv)
X_test, y_test = df_test['text'].tolist(), df_test['label'].values
X_test_emb = embed_texts(X_test)
y_prob = svm_model.predict_proba(X_test_emb)[:,1]
y_pred = (y_prob>best_t).astype(int)

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
print("Metrics:",
        f"Acc={accuracy_score(y_test,y_pred):.4f}",
        f"Prec={precision_score(y_test,y_pred):.4f}",
        f"Rec={recall_score(y_test,y_pred):.4f}",
        f"F1={f1_score(y_test,y_pred):.4f}")
plot_confusion(y_test, y_pred)