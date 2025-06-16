# GAN Training Script

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

# Load and preprocess dataset
df = pd.read_csv('../../dataset.csv')
df.drop(['Filename', 'header'], axis=1, inplace=True)
df = df[df['Class'] == 'Malicious']
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

encoder = LabelEncoder()
df['text'] = encoder.fit_transform(df['text'])
df['Class'] = df['Class'].apply(lambda x: 1 if 'malicious' in str(x).lower() else 0)

categorical_columns = df.select_dtypes(include=['object']).columns

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

for column in categorical_columns:
    valid_integers = [int(x) for x in df[column] if is_integer(x)]
    mode_value = max(set(valid_integers), key=valid_integers.count) if valid_integers else 0
    df[column] = df[column].apply(lambda row: int(row) if is_integer(row) else mode_value)

features = df.drop('Class', axis=1).values
scaler = StandardScaler()
features = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)
dataset = TensorDataset(features)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# GAN Parameters
params = {
    'lr': 0.0002,
    'batch_size': 64,
    'lambda_gp': 8.63,
    'epochs': 100,
    'latent_dim': 200,
    'beta1': 0.5,
    'beta2': 0.999,
}

# Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones_like(d_interpolates),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def mode_collapse_score(samples, n_clusters=3):
    samples_np = samples.detach().cpu().numpy()
    cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(samples_np)
    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
    dist = cluster_counts / len(samples_np)
    ideal = np.ones(n_clusters) / n_clusters
    return np.sum(ideal * np.log(ideal / (dist + 1e-10)))

def check_quality(data_df, generated_df, verbose=False):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data_df)
    return evaluate_quality(data_df, generated_df, metadata, verbose=verbose).get_score()

def plot_tsne(generator, data_loader, latent_dim, trial_id, device):
    generator.eval()
    real_samples = [real[0].cpu().numpy() for real in data_loader][:2000]
    real_np = np.vstack(real_samples)[:2000]
    z = torch.randn(2000, latent_dim).to(device)
    fake_np = generator(z).cpu().numpy()
    combined = np.vstack((real_np, fake_np))
    labels = np.array([0]*2000 + [1]*2000)
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    results = tsne.fit_transform(combined)
    df = pd.DataFrame(results, columns=['x', 'y'])
    df['Label'] = labels
    plt.figure()
    sns.scatterplot(data=df, x='x', y='y', hue='Label', palette='viridis', alpha=0.6)
    plt.title(f"t-SNE Comparison (Trial {trial_id})")
    path = f"tsne_{trial_id}.png"
    plt.savefig(path)
    plt.close()
    return path

def objective(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(params['latent_dim'], features.shape[1]).to(device)
    discriminator = Discriminator(features.shape[1]).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

    g_losses, d_losses, mcs_scores = [], [], []

    for epoch in range(params['epochs']):
        epoch_g_loss, epoch_d_loss = 0, 0
        for i, real in enumerate(data_loader):
            real = real[0].to(device)
            z = torch.randn(real.size(0), params['latent_dim']).to(device)
            fake = generator(z).detach()

            d_real, d_fake = discriminator(real), discriminator(fake)
            gp = compute_gradient_penalty(discriminator, real, fake)
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + params['lambda_gp'] * gp

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            epoch_d_loss += d_loss.item()

            if i % 5 == 0:
                z = torch.randn(real.size(0), params['latent_dim']).to(device)
                fake = generator(z)
                g_loss = -torch.mean(discriminator(fake))
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                epoch_g_loss += g_loss.item()

        g_losses.append(epoch_g_loss / len(data_loader))
        d_losses.append(epoch_d_loss / len(data_loader))
        mcs_scores.append(mode_collapse_score(fake))

    z = torch.randn(2000, params['latent_dim']).to(device)
    generated_np = generator(z).cpu().numpy()
    real_np = np.vstack([real[0].cpu().numpy() for real in data_loader])[:2000]

    real_df = pd.DataFrame(real_np)
    fake_df = pd.DataFrame(generated_np)
    real_df.columns = real_df.columns.astype(str)
    fake_df.columns = fake_df.columns.astype(str)
    quality = check_quality(real_df, fake_df)

    run_id = f"run_{np.random.randint(10000)}"
    os.makedirs(f"models/{run_id}", exist_ok=True)
    torch.save(generator.state_dict(), f"models/{run_id}/generator.pth")
    plot_tsne(generator, data_loader, params['latent_dim'], run_id, device)

    return g_losses, d_losses, mcs_scores, quality

# Execute Training
g_loss, d_loss, mcs, quality = objective(params)

# Plotting
plt.figure()
plt.plot(g_loss, label='Generator Loss')
plt.plot(d_loss, label='Discriminator Loss')
plt.title('Losses Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(mcs, label='Mode Collapse Score', color='green')
plt.title('Mode Collapse Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.show()

print(f"Final Sample Quality Score: {quality:.4f}")
print(f"Final Mode Collapse Score: {mcs[-1]:.4f}")
