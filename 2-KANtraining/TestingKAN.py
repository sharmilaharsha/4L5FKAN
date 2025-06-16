# testing.py
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from KANtraining import KAN, clean_and_convert  #   importable from our module

# Load artifacts
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load data
data = pd.read_csv('/home/sharmila/MyCodes/PDFMalware/OurData.csv')
feature_columns = [ ... ]  # same list as in training

# Preprocess test set
features = data[feature_columns].applymap(clean_and_convert)
X_train, X_test, y_train, y_test = train_test_split(
    features, label_encoder.transform(data['Class']),
    test_size=0.2, random_state=42
)
X_test_pre = preprocessor.transform(X_test)
X_test_tensor = torch.tensor(X_test_pre, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(layers_hidden=[X_test_tensor.shape[1], 128, 64, 32, 2])
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Predictions
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

# Metrics
print(classification_report(y_test, preds, target_names=label_encoder.classes_))
cm = confusion_matrix(y_test, preds)
roc_auc = roc_auc_score((y_test>0).astype(int), probs[:,1])
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC
fpr, tpr, _ = roc_curve((y_test>0).astype(int), probs[:,1])
plt.figure(); plt.plot(fpr, tpr);
plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.show()

# Plot Confusion Matrix
plt.figure(); sns.heatmap(cm, annot=True, fmt='d'); plt.title('Confusion Matrix'); plt.show()
