import numpy as np

features = np.load("tst_features.npy", allow_pickle=True)
targets = np.load("tst_targets.npy", allow_pickle=True)


# Reduce the dimensionality of the features to 50 with PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)


# Plot the features with t-SNE and scatter plot
# The nodes are colored by `targets`

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features_pca)

# Create a bigger plot
plt.figure(figsize=(16, 16))
plt.scatter(
    features_tsne[:, 0],
    features_tsne[:, 1],
    c=targets,
    cmap="tab10",
    alpha=0.7,
)
plt.title("t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# Do the same plot but with 5 samples per class
# The nodes are colored by `targets`
plt.figure(figsize=(16, 16))
n = 10
features_tsne_ = []
targets_ = []
for __i in range(100):
    features_tsne_ += list(features_tsne[targets == __i][:n])
    targets_ += [__i] * n
features_tsne_ = np.array(features_tsne_)
targets_ = np.array(targets_)

plt.scatter(
    features_tsne_[:, 0],
    features_tsne_[:, 1],
    c=targets_,
    cmap="tab10",
    alpha=0.7,
)
plt.title("t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()


# Calculate the mean of the features per class
features_mean = np.array(
    [np.mean(features[targets == __i], axis=0) for __i in range(100)]
)


# Predict the class of the features by calculating the Euclidean distance
# between the features and the mean of the features per class
features_predicted = np.array(
    [
        np.argmin(np.linalg.norm(features_mean - __f, axis=1))
        for __f in features
    ]
)
print(f"Accuracy: {np.mean(features_predicted == targets)}")
