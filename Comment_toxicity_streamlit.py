import nltk
import os

# Set NLTK data path to a directory in your app
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download only stopwords and wordnet (skip punkt completely)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

# Now import rest
import streamlit as st
import pandas as pd
import torch
import json
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch import nn
import urllib.request

# ----- Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Toxicity Detection App")

# ----- Load files from GitHub -----
repo_url = "https://raw.githubusercontent.com/AshvinAK17/Comment-Toxicity/master/"

# Helper: download file if not exists locally
def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

# Download needed files
download_if_not_exists("word2idx.json", repo_url + "word2idx.json")
download_if_not_exists("BiLSTM_metrics.json", repo_url + "BiLSTM_metrics.json")
download_if_not_exists("BiLSTM_model.pth", "https://github.com/AshvinAK17/Comment-Toxicity/raw/master/BiLSTM_model.pth")

# Load word2idx
with open("word2idx.json") as f:
    word2idx = json.load(f)

# Load metrics
metrics = {}
with open("BiLSTM_metrics.json") as f:
    metrics["BiLSTM"] = json.load(f)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
vocab_size = len(word2idx)
emb_dim = 192
hidden_dim = 96
output_dim = len(label_cols)

# ----- BiLSTM Model -----
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(concat)

# Load trained BiLSTM model
models = {}
bilstm_model = BiLSTM(vocab_size, emb_dim, hidden_dim, output_dim).to(device)
bilstm_model.load_state_dict(torch.load("BiLSTM_model.pth", map_location=device))
bilstm_model.eval()
models["BiLSTM"] = bilstm_model

# ----- Text preprocessing -----
sw = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in sw and len(t) >= 4]
    return tokens

def text_to_tensor(tokens):
    if not tokens:
        # add <UNK> if tokens are empty to avoid empty tensor
        indices = [word2idx.get("<UNK>", 0)]
    else:
        indices = [word2idx.get(word, word2idx["<UNK>"]) for word in tokens]
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    return tensor

# ----- Streamlit UI -----
st.title("Comment Toxicity Detection")

tab1, tab2, tab3 = st.tabs(["💬Single Comment", "📈 Metrics & Insights", "📁 Bulk CSV"])

with tab1:
    st.subheader("Enter a comment to check toxicity:")
    user_input = st.text_area("Comment", height=100)
    selected_model = "BiLSTM"
    if st.button("Predict"):
        if user_input.strip():
            tokens = clean_text(user_input)
            tensor = text_to_tensor(tokens)
            with torch.no_grad():
                logits = models[selected_model](tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                binary_preds = (probs > 0.5).astype(int)
            result = {label: int(pred) for label, pred in zip(label_cols, binary_preds)}
            st.write("### Prediction:")
            st.json(result)
        else:
            st.warning("Please enter a comment!")

with tab2:
    st.subheader("📊 Model Performance Metrics")
    st.write("**BiLSTM**")
    st.json(metrics["BiLSTM"])

with tab3:
    st.subheader("📁 Upload CSV for bulk prediction")
    st.caption("⚠️ CSV must have only two columns: `id` and `comment_text`")
    uploaded = st.file_uploader("Upload CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "comment_text" not in df.columns:
            st.error("CSV must have a 'comment_text' column!")
        else:
            all_preds = []
            for text in df['comment_text']:
                tokens = clean_text(str(text))
                tensor = text_to_tensor(tokens)
                with torch.no_grad():
                    logits = models["BiLSTM"](tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                    binary_preds = (probs > 0.5).astype(int)
                all_preds.append(binary_preds)
            pred_df = pd.DataFrame(all_preds, columns=label_cols)
            output = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
            st.write("### Predictions:")
            st.dataframe(output)
            csv = output.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions as CSV", csv, "bulk_predictions.csv", "text/csv")

st.markdown("---")
st.caption(" 🛠️ Built with PyTorch & Streamlit | Ashvin’s Toxicity Detection App")
