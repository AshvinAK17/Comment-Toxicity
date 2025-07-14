import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

# Model classes must be defined/imported:
# TextCNN and BiLSTM (must match training setup)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model hyperparameters
emb_dim = 192
hidden_dim = 96
output_dim = 6  # number of labels
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ✅ Load test_df directly from GitHub
test_df = pd.read_csv("https://media.githubusercontent.com/media/AshvinAK17/Comment-Toxicity/refs/heads/master/test.csv")

# ✅ Load saved word2idx from JSON (must be in the same folder or GitHub clone)
with open("word2idx.json") as f:
    word2idx = json.load(f)

vocab_size = len(word2idx)

# ✅ Vectorize & pad test data
test_vector = []
for text in test_df['comment_text']:
    tokens = str(text).split()
    indices = [word2idx.get(word, word2idx["<UNK>"]) for word in tokens]
    test_vector.append(torch.tensor(indices))
test_padded = pad_sequence(test_vector, batch_first=True, padding_value=0)

# ✅ Prediction + save function (unchanged)
def predict_and_save(model_class, model_name, model_file, test_padded, test_df, vocab_size, emb_dim, output_dim, label_cols):
    test_loader = DataLoader(test_padded, batch_size=64, shuffle=False)

    model = model_class(vocab_size, emb_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())

    pred_test = torch.cat(all_preds, dim=0).numpy()
    pred_df = pd.DataFrame(pred_test, columns=label_cols).astype(int)
    output_df = pd.concat([test_df['comment_text'].reset_index(drop=True), pred_df], axis=1)

    filename = f"{model_name}_test_predictions.csv"
    output_df.to_csv(filename, index=False)
    print(f"✅ Saved predictions to {filename}")

# --- Run for TextCNN ---
predict_and_save(
    model_class=TextCNN,
    model_name="TextCNN",
    model_file="TextCNN_model.pth",
    test_padded=test_padded,
    test_df=test_df,
    vocab_size=vocab_size,
    emb_dim=emb_dim,
    output_dim=output_dim,
    label_cols=label_cols
)

# --- Run for BiLSTM ---
predict_and_save(
    model_class=lambda vocab_size, emb_dim, output_dim: BiLSTM(vocab_size, emb_dim, hidden_dim=hidden_dim, output_dim=output_dim),
    model_name="BiLSTM",
    model_file="BiLSTM_model.pth",
    test_padded=test_padded,
    test_df=test_df,
    vocab_size=vocab_size,
    emb_dim=emb_dim,
    output_dim=output_dim,
    label_cols=label_cols
)
