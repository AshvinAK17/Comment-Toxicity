import pandas as pd
import nltk
import re
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# 1. Setup & Configuration
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
batch_size = 32
emb_dim = 192
hidden_dim = 96
epochs = 15

# 2. Data Preprocessing functions

sw = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def get_clean_text(text):
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
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if (t.isalpha()) and (t not in sw)]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) >= 4]
    return tokens

def preprocess_data(train_path):
    train_df = pd.read_csv(train_path)
    train_df.drop(['id'], axis=1, inplace=True)
    train_df['cleaned'] = train_df['comment_text'].apply(get_clean_text)
    return train_df

# 3. Encoding: word2idx & padding

def create_word2idx(X_train):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for tokens in X_train:
        for word in tokens:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    return word2idx

def vectorize_and_pad(X_data, word2idx):
    vector_list = []
    for sent in X_data:
        inner_vector = [word2idx.get(word, word2idx["<UNK>"]) for word in sent]
        vector_list.append(torch.tensor(inner_vector))
    padded = pad_sequence(vector_list, batch_first=True, padding_value=0)
    return padded

# 4. Compute class weights to handle imbalance

def compute_pos_weight(train_df):
    num_pos = train_df[label_cols].sum(axis=0).values
    num_neg = len(train_df) - num_pos
    return torch.tensor(num_neg / (num_pos + 1e-5), dtype=torch.float32).to(device)

# 5. Model Definitions

# Bidirectional LSTM for text classification.
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
    def forward(self, X):
        X = self.embedding(X)
        output, (hidden, cell_state) = self.lstm(X)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        concatenated = torch.cat((hidden_forward, hidden_backward), dim=1)
        return self.classifier(concatenated)

# CNN with multiple kernel sizes for text classification.
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, output_dim, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, emb_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(pool, dim=2)[0] for pool in x]
        x = torch.cat(x, dim=1)
        return self.fc(self.dropout(x))

# 6. Training & Evaluation

def train_and_evaluate(model, train_loader, pos_weight, input_train, labels):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    print(f"\nTraining model: {type(model).__name__}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    eval_loader = DataLoader(TensorDataset(input_train, labels), batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in eval_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())
    pred_train = torch.cat(all_preds, dim=0).numpy()
    labels_cpu = labels.cpu().numpy()

    print(f"Accuracy: {accuracy_score(labels_cpu, pred_train)}")
    print(f"Precision: {precision_score(labels_cpu, pred_train, average='macro', zero_division=0)}")
    print(f"Recall : {recall_score(labels_cpu, pred_train, average='macro', zero_division=0)}")
    print(f"F1 Score: {f1_score(labels_cpu, pred_train, average='macro', zero_division=0)}")

    model_path = f"{type(model).__name__}_model.pth"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "Accuracy": float(accuracy_score(labels_cpu, pred_train)),
        "Precision": float(precision_score(labels_cpu, pred_train, average='macro', zero_division=0)),
        "Recall": float(recall_score(labels_cpu, pred_train, average='macro', zero_division=0)),
        "F1 Score": float(f1_score(labels_cpu, pred_train, average='macro', zero_division=0))
    }

    metrics_filename = f"{type(model).__name__}_metrics.json"
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_filename}")

# 7. Run script directly: load data, prepare & train

train_path = "https://media.githubusercontent.com/media/AshvinAK17/Comment-Toxicity/refs/heads/master/test.csv"
train_df = preprocess_data(train_path)
word2idx = create_word2idx(train_df['cleaned'])
train_padded = vectorize_and_pad(train_df['cleaned'], word2idx)
train_labels = torch.tensor(train_df[label_cols].values, dtype=torch.float32)

pos_weight = compute_pos_weight(train_df)
dataset = TensorDataset(train_padded, train_labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

vocab_size = len(word2idx)
models = [
    BiLSTM(vocab_size, emb_dim, hidden_dim, len(label_cols)).to(device),
    TextCNN(vocab_size, emb_dim, len(label_cols)).to(device)
]

input_train = train_padded.to(device)
labels = train_labels.to(device)

for model in models:
    train_and_evaluate(model, train_loader, pos_weight, input_train, labels)