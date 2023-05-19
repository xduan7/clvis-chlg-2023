import torch


def _k_means(
    features: torch.Tensor,
    num_clusters: int,
    max_num_iter: int = 100,
    tol: float = 1e-4,
):
    _centers = features[torch.randperm(features.size(0))[:num_clusters]]
    for _ in range(max_num_iter):
        distances = torch.cdist(features, _centers)
        labels = torch.argmin(distances, dim=1)
        _new_centers = torch.stack(
            [features[labels == i].mean(dim=0) for i in range(num_clusters)]
        )
        if torch.norm(_new_centers - _centers) < tol:
            break
        _centers = _new_centers

    # Get the features that are closest to each center
    # _nearest_indices = []
    # for __c in _centers:
    #     __dist = torch.cdist(features, __c.view(1, -1))
    #     _nearest_indices.append(torch.argmin(__dist, dim=0))
    # return (
    #     features[_nearest_indices, :],
    #     torch.zeros_like(features[_nearest_indices, :]),
    # )

    # Get the mean and std of each cluster
    return (
        torch.stack(
            [
                features[labels == __i, :].mean(dim=0)
                for __i in range(num_clusters)
            ]
        ),
        torch.stack(
            [
                features[labels == __i, :].std(dim=0)
                for __i in range(num_clusters)
            ]
        ),
    )


# Make a test feature with two clusters
_num_samples_in_cluster = 100
_num_features = 2
_test_features = torch.cat(
    [
        torch.randn(_num_samples_in_cluster, _num_features) * 2 - 5,
        torch.randn(_num_samples_in_cluster, _num_features) * 4 + 5,
    ]
)

# Run k-means
_num_clusters = 2
_mean, _std = _k_means(_test_features, _num_clusters)
print(_mean)
print(_std)

_logits = []
for __i in range(_mean.shape[0]):
    __l = [
        torch.normal(_mean[__i], _std[__i])
        for _ in range(32)
    ]
    __l = torch.stack(__l, dim=0)
    _logits.append(__l)
    break
_logits = torch.cat(_logits, dim=0)
print(_logits)