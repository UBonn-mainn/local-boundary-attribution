import matplotlib.pyplot as plt
import torch

def plot_proximity_hist(distances):
    plt.hist(distances, bins=20)
    plt.xlabel("Distance to decision boundary")
    plt.ylabel("Count")
    plt.title("Proximity Distribution")
    plt.show()

def plot_disagreement_matrix(matrix, labels):
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title("Attribution Method Disagreement")
    plt.show()

def visualize_attributions(attrs, titles):
    n = len(attrs)
    plt.figure(figsize=(4 * n, 4))
    for i, (a, t) in enumerate(zip(attrs, titles)):
        plt.subplot(1, n, i + 1)
        plt.imshow(a.squeeze().cpu(), cmap="bwr")
        plt.title(t)
        plt.axis("off")
    plt.show()
