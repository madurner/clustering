import numpy as np
#from scipy import misc
import imageio
from time import time
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import cm

import utils

def kmeans_compression(img, K=5):
    num_rows, num_cols, num_channels = img.shape

    # Rearrange image to (row*cols) x channels
    img = np.vstack(img)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    img_compressed = kmeans(img, K=K, max_iter=5, ax1=ax1, ax2=ax2)

    # Reshape labeling to (num_rows, num_cols) and color with the labeling
    img_compressed = img_compressed.reshape((num_rows, num_cols))
    img_compressed = img_compressed.astype('uint8')

    #file_out = img_file + '_K' + str(num_components) + ext
    # plt.imsave(file_out, Y, cmap='gray')
    plt.figure()
    plt.imshow(img_compressed, cmap='gray')
    plt.show()


def kmeans(data, K, max_iter=100, threshold = 1e-2, ax1=None, ax2=None):
    
    """ applies kmeans algorithm 

    Args:
        data (): [nxd] matrix of n samples with d prperties each; 
        K (int): number of clusters
        max_iter (int): maximum amount of cluster iterations
        threshold (float): minimum of mean_shift for continuing cluster process

    Return:
    """
    n, feat_dim = data.shape

    # Initialize cluster means randomly within the data range
    d_min = np.min(data, axis=0)
    d_max = np.max(data, axis=0)
    _range = d_max - d_min
    means = np.array([d_min + np.random.rand(feat_dim)*_range for _ in range(K)])


    # Convergence criteria
    converged = False
    old_cost, old_meanshift = np.infty, 0

    if ax1 is not None:
        ax1.set_title('K-means cost')

    if ax2 is not None:
        ax2.set_title('Mean shift')

    for i in range(max_iter):
        print('Iteration %d' % i)

        assignment, cost = utils.assign_data(data, means)

        # Re-initialize means if a cluster has no samples assigned to it
        while np.unique(assignment).shape[0] != K:
            means = np.array([d_min + np.random.rand(feat_dim)*_range for _ in range(K)])
            assignment, cost = utils.assign_data(data, means)

        # Update cluster-means
        old_means = means.copy()
        for k in range(K):
            idx = np.array(assignment == k)
            means[k, :] = np.mean(data[idx, :], axis=0)

        # Compute the mean shift for each cluster
        sq_mean_shift = np.zeros(K)
        for k in range(K):
            diff = means[k, :] - old_means[k, :]
            #sq_mean_shift[k] = np.dot(diff, diff)
            sq_mean_shift[k] = np.linalg.norm(diff)

        #meanshift = np.linalg.norm(sq_mean_shift)
        meanshift = np.mean(sq_mean_shift)

        # Visualization only
        if ax1 is not None and i > 0:
            dt = [i-1, i]
            ax1.plot(dt, [old_cost, cost], 'b-o')
            #TODO visualize threshol
            ax2.plot(dt, [old_meanshift, meanshift], 'b-o')

            plt.gcf().suptitle('Iter %d' % i)
            plt.draw()
            plt.pause(0.1)

        old_cost = cost
        old_meanshift = meanshift

        # Evaluate converge criterion
        if all([m < threshold for m in sq_mean_shift]):
            converged = True
            print('Converged after ', i, ' iterations')
            break

    if not converged:
        print('Did not converge after ', i, ' iterations')

    return assignment

def main():
    img_file = 'data/mona-lisa.jpg'
    img = imageio.imread(img_file)

    kmeans_compression(img)


if __name__ == '__main__':
    main()

