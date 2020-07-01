import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial.distance import cdist

def assign_data(data, means, dist_type='sqeuclidean'):
    """ assigns each data point to a cluster(-mean)

    Args:
        data (): [nxd] matrix of n samples with d prperties each; 
        means (): [Kxd] matrix of K means;
        dist_type (string): distance type; default: sqeuclidean

    Return:
        assignment:
        cost: sum of distances between each data point and its corresponding cluster-mean
    """
    n, feat_dim = data.shape

    #compute distance between every data point and mean value
    dist = cdist(data, means, dist_type)

    # for each data point: smallest distance -> cluster assignment
    assignment = np.argmin(dist, axis=1)
    cost = np.sum(dist[np.arange(n), assignment])

    return assignment, cost

def compute_random_means(_min, _range, K):
    d = len(_min)
    means = np.array([_min + np.random.rand(d)*_range for _ in range(K)])
    return means

def cond_plot(it, title=None, data=None, covs=None, colors=None, marker='o', art=None, ax=None):
    if art is not None:
        [a.remove() for a in art]
    artists = []
    plt.sca(ax)
    plt.scatter(data[:, 0], data[:, 1], c=colors, marker=marker, lw=0, s=50)
    if covs is not None:
        n_components, n_features = data.shape
        for k in range(n_components):
            ell = plot_ellipse(plt.gca(), data[k, :], covs[k, :, :], color=colors[k])
            artists.append(ell)
    if title is not None:
        plt.title('%s - %s' % (it, title), fontweight='bold')
    plt.draw()
    plt.pause(0.5)
    return artists

def plot_ellipse(ax, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = Ellipse(xy=mean, width=2 * v[0] ** 0.5, height=2 * v[1] ** 0.5, angle=180 + angle,
                    facecolor='lightblue', edgecolor='red', linewidth=2, zorder=2)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    return ell
