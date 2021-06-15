import numpy as np
import torch
import matplotlib.pyplot as plt

from spcdist.torch import (MultivariateBetaGaussian,
                           MultivariateBetaGaussianDiag)


if __name__ == '__main__':

    # generate a few near-uniform blobs.

    n_clusters = 3
    d = 2
    torch.random.manual_seed(1)

    mean = 3 * torch.randn(n_clusters, 2)
    P = torch.randn(n_clusters, 2, 2)
    scales = .5 * torch.eye(2) + .5 * P @ P.transpose(2, 1)
    alpha = torch.tensor(10.0)

    mbg = MultivariateBetaGaussian(mean, scales, alpha=alpha)
    X = mbg.rsample((1000,)).reshape(-1, 2)
    X = X[torch.randperm(X.shape[0]), :]
    X.detach_()

    # now, let's refit from scratch
    loc = torch.randn(size=(n_clusters, d), requires_grad=True)
    scale_diag = torch.randn(size=(n_clusters, d), requires_grad=True)
    cluster_logits = torch.zeros((X.shape[0], n_clusters), requires_grad=True)

    parameters = [loc, scale_diag, cluster_logits]

    # opt = torch.optim.Adam(parameters, lr=.9)
    opt = torch.optim.SGD(parameters, lr=.01)

    n_iter = 10001

    for it in range(n_iter):
        opt.zero_grad()

        scale = scale_diag ** 2
        model = MultivariateBetaGaussianDiag(loc, scale, alpha=alpha)
        dists = model.cross_fy(X, broadcast_batch=True)
        cluster_proba = torch.softmax(cluster_logits, dim=-1)
        loss = torch.mean(torch.sum(dists * cluster_proba), dim=-1)

        if it % 1000 == 0:
            print(loss.item())
        loss.backward()
        opt.step()


    X_np = X.numpy()
    y = torch.min(dists, dim=-1).indices.numpy()
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(X_np[y == i, 0], X_np[y == i, 1], alpha=.2, marker='.')

    X_s = model.rsample((1000,))
    X_s = X_s.detach().numpy()
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(X_s[:, i, 0], X_s[:, i, 1], alpha=.2, marker='.')

    plt.show()
