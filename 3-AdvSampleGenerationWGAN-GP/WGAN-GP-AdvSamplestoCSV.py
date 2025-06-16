# gan_training

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

# Load and preprocess data
df = pd.read_csv('../../dataset.csv')
df.drop(['Filename', 'header'], axis=1, inplace=True)
df = df[df['Class'] == 'Malicious']
df.dropna(inplace=True)
df = df.drop_duplicates()

encoder = LabelEncoder()
df['text'] = encoder.fit_transform(df['text'])
df['Class'] = df['Class'].apply(lambda x: 1 if 'malicious' in str(x).lower() else 0)

# Convert any remaining object-type columns with integers
categorical_columns = df.select_dtypes(include=['object']).columns

def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

for column in categorical_columns:
    valid_integers = [int(x) for x in df[column] if is_integer(x)]
    mode_value = max(set(valid_integers), key=valid_integers.count) if valid_integers else 0
    df[column] = df[column].apply(lambda x: int(x) if is_integer(x) else mode_value)

# Feature preparation
features = df.drop('Class', axis=1).values
scaler = StandardScaler()
features = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)

dataset = TensorDataset(features)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameters
params = {
    'lr': 0.0002,
    'batch_size': 64,
    'lambda_gp': 8.63,
    'epochs': 100,
    'latent_dim': 200,
    'beta1': 0.5,
    'beta2': 0.999,
}

# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Mode Collapse Score
def mode_collapse_score(samples, n_clusters=3):
    samples_np = samples.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(samples_np)
    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
    cluster_distribution = cluster_counts / len(samples_np)
    ideal_distribution = np.ones(n_clusters) / n_clusters
    mcs = np.sum(ideal_distribution * np.log(ideal_distribution / (cluster_distribution + 1e-10)))
    return mcs

# Quality Check
def check_quality(data_df, generated_df, verbose):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data_df)
    quality_report = evaluate_quality(data_df, generated_df, metadata, verbose=verbose)
    return quality_report.get_score()

# Training Function
def train_gan(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = features.shape[1]
    generator = Generator(params['latent_dim'], input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

    generator_losses, discriminator_losses, mode_collapse_scores = [], [], []

    for epoch in range(params['epochs']):
        loss_G_epoch, loss_D_epoch = 0, 0
        for i, (real_batch,) in enumerate(data_loader):
            real_samples = real_batch.to(device)
            batch_size = real_samples.size(0)

            z = torch.randn(batch_size, params['latent_dim']).to(device)
            fake_samples = generator(z).detach()

            d_real = discriminator(real_samples)
            d_fake = discriminator(fake_samples)
            gp = compute_gradient_penalty(discriminator, real_samples, fake_samples)
            loss_D = -torch.mean(d_real) + torch.mean(d_fake) + params['lambda_gp'] * gp

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            loss_D_epoch += loss_D.item()

            if i % 5 == 0:
                z = torch.randn(batch_size, params['latent_dim']).to(device)
                fake_samples = generator(z)
                loss_G = -torch.mean(discriminator(fake_samples))

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                loss_G_epoch += loss_G.item()

        generator_losses.append(loss_G_epoch / len(data_loader))
        discriminator_losses.append(loss_D_epoch / len(data_loader))
        mcs = mode_collapse_score(fake_samples, n_clusters=3)
        mode_collapse_scores.append(mcs)

        print(f"[Epoch {epoch+1}/{params['epochs']}] "
              f"G Loss: {generator_losses[-1]:.4f} | D Loss: {discriminator_losses[-1]:.4f} | MCS: {mcs:.4f}")

    return generator, generator_losses, discriminator_losses, mode_collapse_scores

# Train
generator, g_loss, d_loss, mcs = train_gan(params)

# Generate 2000 samples and save
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    z = torch.randn(2000, params['latent_dim']).to(device)
    generated_samples = generator(z).cpu().numpy()

# Inverse transform to original scale
generated_samples_original = scaler.inverse_transform(generated_samples)
generated_df = pd.DataFrame(generated_samples_original)
generated_df.columns = df.drop('Class', axis=1).columns

# Save to CSV
output_csv_path = "adv_samples.csv"
generated_df.to_csv(output_csv_path, index=False)
print(f"\n Generated samples saved to {output_csv_path}")

