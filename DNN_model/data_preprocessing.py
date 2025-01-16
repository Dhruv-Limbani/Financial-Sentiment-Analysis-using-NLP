import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle

def load_and_preprocess_data(file_path, val_size=0.1):
    # Load dataset
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    # Label encoding
    label_encoder = LabelEncoder()
    df['Sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])

    # Splitting dataset into features and labels
    x = df[['cleaned_text', 'text_length', 'avg_word_length', 'document_length']]
    y = df['Sentiment_encoded']
    
    # TF-IDF transformation on text data
    tfidf = TfidfVectorizer(max_features=10000)
    x_tfidf = tfidf.fit_transform(x['cleaned_text'])
    
    with open(r'models_and_resources\tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    # Combining TF-IDF features with other numerical features
    feature_matrix = np.hstack([x_tfidf.toarray(), x[['text_length', 'avg_word_length', 'document_length']].values])

    # One-hot encoding labels
    y_cat = to_categorical(y)
    
    # Split data into training and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(
        feature_matrix, y_cat, test_size=val_size + 0.2, random_state=42, stratify=y
    )

    # Further split the temp data into validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return x_train, x_val, x_test, y_train, y_val, y_test, label_encoder, tfidf

def extract_features(input_df):
    input_df['text_length'] = input_df['cleaned_text'].apply(len)
    input_df['avg_word_length'] = input_df['cleaned_text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
    input_df['document_length'] = input_df['cleaned_text'].apply(lambda x: len(x.split()))

    return input_df
