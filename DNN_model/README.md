# Financial Sentiment Analysis using DNN (Dense Neural Network)

## Results Summary:
   ![Classification Report](outputs_and_reports\DNN_classification_report.png)

   ![Output](outputs_and_reports\Output.png)
## Directory Structure
```
.
├── data
│   ├── data.csv                       # Raw dataset
│   ├── Preprocessed_Data.csv          # Preprocessed dataset
├── models_and_resources
│   ├── best_DNN_sentiment_model.pth   # Trained DNN model weights
│   ├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── outputs_and_reports
│   ├── DNN_classification_report.txt  # Classification report
│   ├── Output.png                     # Classification output
├── data_preprocessing.py              # Script for data cleaning and preprocessing
├── model.py                           # DNN model definition
├── predict_sentiment.py               # Script for predicting sentiment from text
├── train.py                           # Training script for the model
├── utils.py                           # Utility functions for evaluation and metrics
└── README.md                          # Project documentation
```

## Overview of Folders and Files

### **Folders**
1. **`data/`**:
   - Contains the raw dataset (`data.csv`) and the preprocessed dataset (`Preprocessed_Data.csv`).

2. **`models_and_resources/`**:
   - Stores trained model weights (`best_DNN_sentiment_model.pth`) for inference.
   - Includes the saved TF-IDF vectorizer (`tfidf_vectorizer.pkl`) to ensure consistent preprocessing during prediction.

3. **`outputs_and_reports/`**:
   - Contains the classification report (`DNN_classification_report.txt`) summarizing model performance.
   - Includes classification results of few financial statements (`Output.png`).

### **Files**
1. **`data_preprocessing.py`**:
   - Handles data cleaning, missing value imputation, label encoding, and feature extraction (e.g., TF-IDF transformation).

2. **`model.py`**:
   - Defines the architecture of the Deep Neural Network (DNN), including input layers, hidden layers, batch normalization, and dropout.

3. **`train.py`**:
   - Manages the training process, including data splitting, early stopping, and saving the best-performing model.

4. **`predict_sentiment.py`**:
   - Takes preprocessed text as input and predicts its sentiment using the trained DNN model.

5. **`utils.py`**:
   - Provides utility functions for evaluating the model, including accuracy computation and generating classification reports.

6. **`README.md`**:
   - Project documentation providing an overview, usage instructions, and details about the directory structure.

## How to Use the Project

### **1. Setting Up the Environment**
1. Clone the repository

2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

### **2. Customization (Optional)**
- Modify hyperparameters, model architecture, or preprocessing steps in the respective files:
  - Hyperparameters: `train.py`
  - Model: `model.py`
  - Preprocessing: `data_preprocessing.py`

### **3. Define and Train the Model (Optional)**
1. You can change the model definition in `model.py`.
2. And execute the following command to train the model:

   ```bash
   python train.py
   ```
   - The script will save the best-performing model in the `models_and_resources/` folder as `best_DNN_sentiment_model.pth`.

### **4. Predict Sentiment**
1. Use the `predict_sentiment.py` script to predict sentiment for custom text inputs:

   ```bash
   python predict_sentiment.py --text "Your sample text here"
   ```
   - Example output:
     ```
     The sentiment of the given sentence is: Positive
     ```

### **4. Visualize Outputs**
1. Check the `outputs_and_reports/` folder for:
   - **Classification Report**: `DNN_classification_report.png`

   ![Classification Report](outputs_and_reports\DNN_classification_report.png)

   - **Classification results of few financial statements**: `Output.png`

   ![Output](outputs_and_reports\Output.png)

### **6. Testing**
- To validate predictions on a batch of data, extend the `predict_sentiment.py` script or integrate it with `utils.py` for batch inference.

---

