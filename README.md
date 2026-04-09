# Detecting-LLM-Generated-Social-Media-Accounts-Using-Semantic-and-Textual
**Members:** Yuexuan He, Zijiang Zhao, Shichen Ma


## ⚙️ Environment
* **Language:** Python 3
* **Necessary Libraries:** `matplotlib`, `pandas`, `numpy`, `scikit-learn`



## 🎯 Core Objective
To build a robust NLP classifier capable of distinguishing between human-written and AI-generated text ($Y \in \{0,1\}$), moving beyond easily manipulated metadata to focus entirely on textual and semantic features.

## 📊 Data Summarization
* **Baseline Training Dataset:** ~6,000 samples (balanced).
   **Source:** From kaggle: hasanyiitakbulut/ai-and-human-text-dataset

* **Evaluation Dataset:** 1,460 samples (highly imbalanced: 1375 Human, 85 AI) used specifically to test for cross-domain generalization.
  **Source:** From kaggle: denvermagtibay/ai-generated-essays-dataset

## 🛠️ Baseline

### 1. The Baseline (Lexical Approach)
* **Architecture:** `TfidfVectorizer` paired with `LogisticRegression`.
* **Result:** Achieved: ....
* **Conclusion:** The model suffered from extreme **overfitting and domain shift**. TF-IDF memorized specific vocabulary (lexical artifacts) from the training set and failed when encountering new text distributions.

### 2. Baseline Optimization
To mitigate the overfitting and severe False Positive rate observed in the baseline:
* **Grid Search:** Applied `GridSearchCV` to constrain the TF-IDF vocabulary (`max_features`, `min_df`) and increase regularization.



## 👉Next Steps:

### The Advanced Model (Semantic Approach...tec)
* **Architecture:** TBD
* **Purpose:** TBD