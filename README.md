###  Comment Toxicity Detection
An interactive deep learning project built with Python and Streamlit that helps identify and flag toxic comments (e.g., hate speech, threats, insults) in text data.  
Users can test single comments, upload CSV files for bulk predictions, and view data insights in a clean web interface.

---

##  Project Objectives
- Classify comments into categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
- Provide real-time predictions for single comments or bulk CSV uploads.
- Build an easy-to-use **Streamlit** web interface for interactive testing.

---

##  Data Source
- `train.csv` and `test.csv` datasets stored in the repository.
- Data originally sourced for comment toxicity classification tasks.

---

##  Data Processing
- **Text cleaning**: lowercasing, removing special characters & stopwords.
- **Tokenization & lemmatization** (using NLTK).
- **Encoding**: building `word2idx` vocabulary.
- **Padding**: to handle different text lengths.

---

##  Technologies Used
- **Python**
- **PyTorch** (deep learning framework)
- **NLTK** (text preprocessing)
- **pandas** (data handling)
- **scikit-learn** (metrics)
- **Streamlit** (web app)

---

##  Used Models
- **BiLSTM**: Bidirectional LSTM for sequence modeling.
- **TextCNN**: Convolutional Neural Network for text classification.

Both models are trained and saved as:
- `BiLSTM_model.pth`
- `TextCNN_model.pth`

---

##  Basic Workflow & Execution
1. **Training** (`Comment-Toxicity.py`):  
   - Cleans and vectorizes data.
   - Trains both models.
   - Saves: model weights, metrics JSON files, and `word2idx.json`.

2. **Testing** (`test_predict.py`):  
   - Loads test data and trained models.
   - Predicts toxic labels.
   - Saves prediction CSV files.

3. **Streamlit App** (`Comment_toxicity_streamlit.py`):  
   - Single comment prediction.
   - Displays metrics.
   - Bulk CSV prediction and allows CSV download.

---

## Project Structure 

```
Comment-Toxicity/
├─ .gitattributes
├─ BiLSTM_metrics.json
├─ BiLSTM_model.pth
├─ BiLSTM_test_predictions.csv
├─ Comment-Toxicity.py
├─ Comment_toxicity_streamlit.py
├─ TextCNN_metrics.json
├─ TextCNN_model.pth
├─ TextCNN_test_predictions.csv
├─ requirements.txt
├─ test.csv
├─ test_predict.py
├─ test_predictions_with_text.csv
├─ train.csv
├─ word2idx.json
└─ streamlit/
```

train.csv / test.csv – Input datasets.

Comment-Toxicity.py – Source code for training and preprocessing.

Comment_toxicity_streamlit.py – Streamlit web application.

BiLSTM_model.pth / TextCNN_model.pth – Saved trained models.

BiLSTM_metrics.json / TextCNN_metrics.json – Saved model performance metrics.

word2idx.json – Vocabulary index mapping.

BiLSTM_test_predictions.csv / TextCNN_test_predictions.csv – Model test prediction outputs.

requirements.txt – Python dependencies.

streamlit/ – (Optional) folder for extra Streamlit components or scripts.

## Streamlit

Direct Link to access the streamlit: **https://ashvinak17-comment-toxicity-comment-toxicity-streamlit-4p5yth.streamlit.app/**

# Features:

- Predict single comment toxicity.

- Upload CSV for bulk predictions.

- View and download prediction results.

# Deployment Guide:

1. **Clone the repository** :

       git clone https://github.com/AshvinAK17/Comment-Toxicity.git
       cd Comment-Toxicity
   
2. **Install dependencies**:

        pip install -r requirements.txt
    
3. **Make sure these files are present** :

 - train.csv
 - test.csv
 - BiLSTM_model.pth
 - TextCNN_model.pth
 - word2idx.json

4. **Run Streamlit web app** :

         python -m streamlit run Comment_toxicity_streamlit.py

The app will open in your browser at http://localhost:8501

