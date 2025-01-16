import torch
import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, x_test, y_test, label_encoder):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(x_test, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)

    y_test_decoded = np.argmax(y_test, axis=1)
    print(classification_report(y_test_decoded, predicted.numpy(), target_names=label_encoder.classes_))