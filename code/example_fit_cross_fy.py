import numpy as np
import torch
import matplotlib.pyplot as plt

from torch_dist import (MultivariateBetaGaussian,
                        MultivariateBetaGaussianDiag)


def fit_mixture():

    # generate a few near-uniform blobs.
    # we can use our own code for this!

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

    # reconstitute best cluster assignment by FY loss.
    dists = mbg.cross_fy(X, broadcast_batch=True)
    y = torch.min(dists, dim=-1).indices.numpy()

    X_np = X.numpy()
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(X_np[y == i, 0], X_np[y == i, 1], alpha=.2, marker='.')
    plt.show()

    # now, let's refit from scratch
    loc = torch.randn(size=(n_clusters, d), requires_grad=True)
    scale_diag = torch.randn(size=(n_clusters, d), requires_grad=True)
    cluster_logits = torch.zeros((X.shape[0], n_clusters), requires_grad=True)

    parameters = [loc, scale_diag, cluster_logits]

    opt = torch.optim.Adam(parameters, lr=.001)

    n_iter = 30001

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



def fit_single():

    torch.manual_seed(42)
    torch.set_default_dtype(torch.double)

    n_samples = 100
    d = 2

    true_loc = torch.tensor([0.0, 0.0])
    true_sigma = torch.tensor([.5, 2.0])

    loc = torch.randn(d, requires_grad=True)
    scale_diag = torch.randn(d, requires_grad=True)

    alpha = torch.tensor(1.25, requires_grad=True)

    parameters = [loc, scale_diag]

    opt = torch.optim.Adam(parameters, lr=1.0)

    n_iter = 10000

    for it in range(n_iter):
        opt.zero_grad()

        scale = scale_diag ** 2
        mbg = MultivariateBetaGaussianDiag(loc, scale, alpha=alpha)
        # get a sample
        X = true_loc + torch.randn((n_samples, d)) * true_sigma

        loss = mbg.cross_fy(X).mean()
        if it % 1000 == 0:
            print(loss.item())
        loss.backward()
        opt.step()

    print("true", true_loc, true_sigma)
    print("learned", loc, scale_diag)

    plt.figure()

    n_samples = 10000

    # true sample
    X = true_loc + torch.randn((n_samples, d)) * true_sigma
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(),
                marker='.', alpha=.2,
                label="true")

    Z = mbg.rsample(sample_shape=(n_samples,)).detach()
    plt.scatter(Z[:, 0].numpy(), Z[:, 1].numpy(),
                marker='.', alpha=.2,
                label="pred")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # fit_single()
    fit_mixture()

