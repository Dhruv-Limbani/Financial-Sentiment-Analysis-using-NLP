## Financial Sentiment Analysis Project Overview  

This project involves analyzing and modeling a sentiment-labeled text dataset through preprocessing, feature engineering, and applying machine learning and deep learning models. The work is divided into multiple notebooks and folders, each addressing specific steps in the pipeline.  

---

### **1. Text Data Preprocessing and Analysis (Data Analysis.ipynb)**  
This notebook demonstrates text preprocessing and exploratory data analysis on the dataset. Key steps include:  

#### **1.1 Data Overview and Initial Exploration**  
- Inspect the dataset structure and the first few rows.  
- Analyze text length distribution and sentiment class distribution.  

#### **1.2 Text Cleaning**  
- Convert text to lowercase and remove special characters.  
- Tokenize and lemmatize words while removing stopwords.  

#### **1.3 Text Length and Word Analysis**  
- Calculate and visualize the distribution of text lengths.  
- Analyze average word lengths and compare distributions across sentiment classes.  

#### **1.4 N-gram Analysis**  
- Generate bigrams and trigrams from the cleaned text.  
- Visualize the most common bigrams and trigrams for each sentiment class.  

#### **1.5 TF-IDF Analysis**  
- Apply TF-IDF vectorization to the text.  
- Visualize the top 20 words by average TF-IDF scores for each sentiment class.  

#### **1.6 Box Plots**  
- Display box plots for text length and average word length by sentiment class.  

#### **1.7 Saving Processed Data**  
- Save the preprocessed dataset as a CSV file for downstream modeling.  

---

### **2. Deep Learning Models**  

#### **2.1 BiLSTM in PyTorch `BiLSTM_pytorch.ipynb`**  
- A BiLSTM model was built using PyTorch with the following features:  
  - Bidirectional LSTM with two layers.  
  - Embedding layer and dropout for regularization.  
  - Final linear layer for classification.  
- Achieved **88% test accuracy**.  

#### **2.2 BiLSTM in TensorFlow `BiLSTM_tensorflow.ipynb`**  
- Implemented a BiLSTM model using TensorFlow with similar architecture and hyperparameters as the PyTorch version.  
- Achieved **90.5% test accuracy**.  

#### **2.3 DNN Model `DNN/README.md`**  
- Built a simple Deep Neural Network (DNN) for sentiment classification.  
- Key features:  
  - Dense layers with ReLU activation and dropout regularization.  
  - Final output layer with softmax for classification.  
- Achieved **65% accuracy**.  
- check `README.md` file under `DNN` folder for more details.

---

### **3. Classical Machine Learning Models `ML_algos_based_sentiment_classifier.ipynb`**  
- Applied classical ML algorithms to classify sentiment after preprocessing.  
- Performed hyperparameter tuning to optimize each algorithm.  
- **Results**:  
  - SVM: **~70% accuracy** (highest among ML models).  
  - Random Forest and Logistic Regression performed below SVM.  

---

### **4. Hybrid and Advanced Deep Learning Models `BERT_and_LSTM.ipynb`**  

#### **4.1 BERT-Based Classifier**  
- Fine-tuned a BERT model for sentiment classification.  
- Achieved **70.4% accuracy**.  

#### **4.2 LSTM Classifier**  
- Implemented a standard LSTM model without bidirectionality or advanced layers.  
- Achieved **63% accuracy**.  

---

### **5. Results Summary**  

| **Model**                  | **Framework**  | **Accuracy** |  
|----------------------------|----------------|--------------|  
| BiLSTM                     | TensorFlow     | **90.5%**    |  
| BiLSTM                     | PyTorch        | **88%**      |  
| BERT-Based Classifier      | HuggingFace    | 70.4%        |  
| SVM                        | Scikit-learn   | ~70%         |  
| DNN                        | TensorFlow     | 65%          |  
| LSTM                       | TensorFlow     | 63%          |  

---