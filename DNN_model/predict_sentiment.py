# Suppress TensorFlow logs and warnings
import warnings
import os
import sys
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Redirect standard error to null to suppress additional warnings
sys.stderr = open(os.devnull, 'w')

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import extract_features
from model import DenseModel
import pickle

# Load model and dependencies
model = DenseModel(input_dim=10003, num_classes=3)  
model.load_state_dict(torch.load(r'models_and_resources\best_DNN_sentiment_model.pth'))
model.eval()

label_encoder = LabelEncoder()
label_encoder.fit(['negative', 'neutral', 'positive'])  

with open(r'models_and_resources\tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Define preprocessing function
def preprocess_input(text):
    input_df = pd.DataFrame({'cleaned_text': [text]})
    input_df = extract_features(input_df)

    tfidf_features = tfidf.transform(input_df['cleaned_text']).toarray()
    
    feature_matrix = np.hstack([
        tfidf_features,
        input_df[['text_length', 'avg_word_length', 'document_length']].values
    ])

    return torch.tensor(feature_matrix, dtype=torch.float32)

if __name__ == '__main__':
    # Restore standard error for the main part of the script
    sys.stderr = sys.__stderr__

    input_text = sys.argv[1]
    input_tensor = preprocess_input(input_text)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    sentiment = label_encoder.inverse_transform([predicted_class.item()])[0]
    print(f"The sentiment of the given sentence is: {sentiment}")
