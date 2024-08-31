"""
Inspired by 3b1b's video on LLM explainability
https://www.youtube.com/watch?v=9-Jl0dxWQs8

Johnson-Lindenstrauss lemma states that a number of high-dimensional points can be embedded into a lower-dimensional
space in such a way that the distances between the points are nearly preserved.

O(log(m) / epsilon^2) dimensions are enough to preserve the distances between the points with a factor of (1 +/- epsilon)
"""
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


NUM_EMBEDDINGS = 10000  # N
EMBEDDING_DIM = 50  # D
DOT_DIFF_CUTOFF = 0.01
NUM_STEPS = 250
LR = 0.01
PLOT_EVERY = 25

DEVICE = torch.device("mps")
torch.manual_seed(42)
# torch.set_default_dtype(torch.float64)

embedding_matrix = torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM, device=DEVICE)  # N x D
embedding_matrix /= embedding_matrix.norm(p=2, dim=1, keepdim=True)  # Normalize
embedding_matrix.requires_grad_(True)

# Create nearly-perpendicular vectors with optimizer
optimizer = torch.optim.Adam([embedding_matrix], lr=LR)
identity_matrix = torch.eye(NUM_EMBEDDINGS, NUM_EMBEDDINGS, device=DEVICE)
losses = []


def plot_loss():
    # Plot the loss over time
    plt.plot(losses)
    plt.grid(1)
    plt.show()


def plot_angles():
    # Plot the angle distribution
    dot_products = embedding_matrix @ embedding_matrix.T
    norms = torch.sqrt(torch.diag(dot_products))
    normed_dot_products = dot_products / torch.outer(norms, norms)  # Normalize
    angles_deg = torch.rad2deg(torch.acos(normed_dot_products.detach()))
    # Remove self-orthogonality
    non_diag_orthogonality_mask = ~(torch.eye(NUM_EMBEDDINGS, NUM_EMBEDDINGS, device=DEVICE).bool())

    angles_cpu = angles_deg[non_diag_orthogonality_mask].cpu()
    plt.hist(angles_cpu, bins=1000, range=(0, 180))
    plt.grid(1)
    plt.show()


plot_angles()

for step in tqdm(range(NUM_STEPS)):
    optimizer.zero_grad()

    # Find the dot product of all pairs of vectors
    dot_products = embedding_matrix @ embedding_matrix.T  # N x N

    # Subtract the identity matrix to get rid of the dot product of a vector with itself
    # This will make sure we minimize the dot product of all other vectors
    non_diag_dot_products = dot_products - identity_matrix

    # This will make sure the dot product of all other vectors is close to 0
    loss = (non_diag_dot_products.abs() - DOT_DIFF_CUTOFF).relu().sum()

    # Extra loss to make sure the vectors are normalized
    loss += NUM_EMBEDDINGS * non_diag_dot_products.diag().pow(2).sum()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if step % PLOT_EVERY == 0:
        plot_angles()

plot_loss()
