# training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ---------- Model Definitions ----------
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=3, spline_order=3,
                 scale_noise=0.05, scale_base=1.0, scale_spline=0.5,
                 enable_standalone_scale_spline=True, base_activation=None,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        # ... initialize buffers and parameters as before
        # (Truncated for brevity; copy your full KANLinear implementation here)
        # Remember to call self.reset_parameters() at end

    # ... include b_splines, curve2coeff, forward, regularization_loss

class KAN(nn.Module):
    def __init__(self, layers_hidden, **kwargs):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_f, out_f, **kwargs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=0.01, regularize_entropy=0.01):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy)
                   for layer in self.layers)

# ---------- Data Loading & Preprocessing ----------
data = pd.read_csv('/home/sharmila/MyCodes/PDFMalware/OurData.csv')

# Define your feature column names list
feature_columns = [ ... ]  # <-- replace with actual feature names as given in feature doc

def clean_and_convert(val):
    if isinstance(val, str):
        val_low = val.lower()
        if val_low in ['yes', 'no']:
            return val_low
        try:
            return str(int(val.split('(')[0]))
        except:
            return val
    return str(val)

# Clean features
features = data[feature_columns].applymap(clean_and_convert)

# Setup preprocessor
categorical_features = feature_columns
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_features)
])

# Encode target
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(data['Class'])
# Binarize if multiclass
if len(np.unique(target)) > 2:
    target = np.where(target > 0, 1, target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
preprocessor.fit(X_train)

# Save preprocessor and label encoder
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Transform and tensorize
X_train_pre = preprocessor.transform(X_train)
X_test_pre = preprocessor.transform(X_test)
X_train_tensor = torch.tensor(X_train_pre, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# ---------- Training Function ----------
def train_model(model, X_tr, y_tr, X_val, y_val,
                num_epochs=100, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    best_val = float('inf'); counter = 0; patience = 10
    for epoch in range(num_epochs):
        # training loop (omitted for brevity)
        pass
    # load best and return
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# ---------- K-Fold Cross-Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_tensor)):
    X_tr_fold = X_train_tensor[tr_idx]
    y_tr_fold = y_train_tensor[tr_idx]
    X_val_fold = X_train_tensor[val_idx]
    y_val_fold = y_train_tensor[val_idx]
    model = KAN(layers_hidden=[X_train_tensor.shape[1], 128, 64, 32, 2])
    model = train_model(model, X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)

# Save final best model
torch.save(model.state_dict(), 'best_model.pth')


