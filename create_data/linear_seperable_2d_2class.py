import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed based on current time for variation
np.random.seed(int(datetime.now().timestamp()))

# Parameters
n_samples = 200  # Number of samples per class
n_features = 2   # 2D data

# Generate Class 0
mean_class0 = np.random.uniform(-5, 0, size=n_features)
cov_class0 = np.eye(n_features) * np.random.uniform(0.5, 1.5)
X_class0 = np.random.multivariate_normal(mean_class0, cov_class0, n_samples)
y_class0 = np.zeros(n_samples)

# Generate Class 1 (separated from Class 0)
mean_class1 = np.random.uniform(2, 7, size=n_features)
cov_class1 = np.eye(n_features) * np.random.uniform(0.5, 1.5)
X_class1 = np.random.multivariate_normal(mean_class1, cov_class1, n_samples)
y_class1 = np.ones(n_samples)

# Combine the data
X = np.vstack((X_class0, X_class1))
y = np.hstack((y_class0, y_class1))

# Shuffle the data
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Save to CSV file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"linearly_separable_data_{timestamp}.csv"

# Combine X and y for saving
data_to_save = np.column_stack((X, y))
np.savetxt(filename, data_to_save, delimiter=',',
           header='feature1,feature2,class', comments='')

print(f"Data saved to: {filename}")
print(f"Total samples: {len(X)}")
print(f"Class 0 samples: {np.sum(y == 0)}")
print(f"Class 1 samples: {np.sum(y == 1)}")

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue',
            label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red',
            label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly Separable 2D Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plot_filename = f"linearly_separable_plot_{timestamp}.png"
plt.savefig(plot_filename, dpi=150)
print(f"Plot saved to: {plot_filename}")

plt.show()
