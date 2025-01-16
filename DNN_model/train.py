# Suppress TensorFlow logs and warnings
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import load_and_preprocess_data
from model import create_model
from sklearn.metrics import accuracy_score
from utils import evaluate_model

# Load the data
x_train, x_val, x_test, y_train, y_val, y_test, label_encoder, tfidf = load_and_preprocess_data(r"data\Preprocessed_Data.csv")

# Create DataLoader for training and validation
train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, optimizer, and loss function
input_dim = x_train.shape[1]
num_classes = y_train.shape[1]
model, optimizer = create_model(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()

num_epochs = 200

# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement in validation loss
best_val_loss = float('inf')
epochs_no_improve = 0

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Training loop
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
        for batch in tepoch:
            inputs, labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels.argmax(dim=1)).sum().item()
            total_predictions += labels.size(0)

            epoch_loss = running_loss / (tepoch.n + 1)
            epoch_accuracy = correct_predictions / total_predictions
            tepoch.set_postfix(loss=epoch_loss, accuracy=epoch_accuracy)

    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

    # Validation after each epoch
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad(): 
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == labels.argmax(dim=1)).sum().item()
            val_total_predictions += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = val_correct_predictions / val_total_predictions
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the model if validation loss improved
        torch.save(model.state_dict(), r'models_and_resources\best_DNN_sentiment_model.pth')
        print(f"Validation loss improved, saving model...")
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
        break

model.load_state_dict(torch.load(r'models_and_resources\best_DNN_sentiment_model.pth'))
model.eval()

print("\n")
evaluate_model(model, x_test, y_test, label_encoder)