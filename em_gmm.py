import matplotlib.pyplot as plt
import numpy as np
#from ex_em_for_gmm import em_gmm
#from clustering_eval import evaluate
from scipy import stats
from matplotlib import cm
import utils


def compute_responsibilities(data, means, covs, pi):
    """ Expectation step
    
        γ(z_k) = p(z_k=1|x) = \frac{p(x|z_k=1)(p(z_k=1)}{\sum_j p(z|z_j=1)p(z_j=1)}
        where p(x|z_k=1) = gaussian N(x|μ_k, Σ_k) and p(z_k=1) = π_k

    Args:
        data: [nxd] data matrix
        means: [Kxd] current means of components
        covs: [Kxdxd] current covariances of components
        pi: [K] current mixture coefficients vector

    Return: 
       y:  responsibilities of size n x K
       llh: joint log likelihood
    """

    K = len(pi)
    n, d = np.shape(data)

    gamma = np.zeros(shape=(n, K))
    for k in range(K): #iterate over clusters
        # compute gaussian
        mvn = stats.multivariate_normal.pdf(data, means[k, :], covs[k, :, :], allow_singular=False)

        gamma[:, k] = pi[k] * mvn

    # Normalize responsibilities
    normalizer = np.sum(gamma, axis=1)
    
    for k in range(K):
        gamma[:, k] /= normalizer

    #log likelihodd
    llh = np.sum(np.log(normalizer))

    return gamma, llh

def update_mixture_params(data, gamma):
    """ Maximization step -> updating model parameters
    
    Args:
        gamma: responsibilities computed in the E-step
        data: data matrix d x n
    Return:
        means, covariances and mixture coefficients pi_k
    """
    n, K = np.shape(gamma)
    n, d = np.shape(data)

    means = np.zeros(shape=(K, d))
    covs = np.zeros(shape=(K, d, d))
    pi = np.zeros(K)

    for k in range(K): #iterate over clusters
        # Compute normalizer for model k
        N_k = np.sum(gamma[:, k])  # 1 x 1

        # Update mixing coefficient of model k
        pi[k] = N_k / n

        # Update mean of model k
        means[k, :] =  np.dot(gamma[:, k].T, data) / N_k # 1 x d

        # Update covariance of model k
        X_k = data - means[None, k, :]  # n x d
        covs[k, :, :] = np.dot(X_k.T, np.dot(np.diag(gamma[:,k]), X_k)) / N_k

    return means, covs, pi

def em_gmm( X, K, max_iter=100, threshold=1e-2, plot=False):
    """ Expectation-Maximization algorithm for Gaussian Mixture Models

    Args:
        X : nxd matrix, the data: n samples with d properties each
        K : number of (gaussian) components to assign the data to
        
    Returns:
        gamma : nxk matrix, the responsibilities
        means : Kxd matrix, the means
        covs  : Kxdxd array, the covariances
        pie   : the K mixing coefficients
        llh   : the final log-likelihood
    """

    n, d = X.shape
    d_min = np.min(X, axis=0)
    d_max = np.max(X, axis=0)
    _range = d_max - d_min

    if plot:
        cmap = cm.get_cmap('jet')
        ccolors = np.array([cmap(float(a + 1) / K) for a in range(K)])
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('Expectation Maximization')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')
        plt.title('Log-likelihood Maximization', fontweight='bold')
        means_marker = '*'

    np.random.seed(1234)
    restart = True
    while restart:
        # Initialize means randomly within the data range
        means = utils.compute_random_means(d_min, _range, K)
        covs = np.array([np.eye(d) for _ in range(K)])
        pi = np.ones(K) / K
        assignment, _ = utils.assign_data(X, means)

        if plot:
            rgba_colors = np.array([cmap(float(a + 1) / K) for a in assignment])
            utils.cond_plot(it='0', data=X, colors=rgba_colors, ax=ax1)
            art = utils.cond_plot(it='0', data=means, covs=covs, colors=ccolors, marker=means_marker, ax=ax1)

        # Convergence criteria
        converged = False

        # Initialize log-likelihood
        llhs = [-np.inf]
        restart = False

        for i in range(max_iter):
            # Store log-likelihood
            llh_prev = llhs[-1]
            print('\nEM iteration %d' % (i))

            # E-step: Compute responsibilities based on new means, covs and pie
            try:
                gamma, llh = compute_responsibilities(X, means, covs, pi)
            except np.linalg.LinAlgError:
                print('Singular covariance detected, re-initializing components randomly...')
                restart = True
                break

            # Visualize the new responsibilities
            if plot:
                assignment = np.argmax(gamma, axis=1)
                confidence = gamma[np.arange(n), assignment]
                rgba_colors = np.array([cmap(float(a+1)/K) for a in assignment])
                rgba_colors[:, -1] = confidence
                # print(rgba_colors)
                utils.cond_plot(it=i, title='E-step (responsibilities)', data=X, colors=rgba_colors, ax=ax1)

            if i > 0:
                llhs.append(llh)
                ax2.plot(llhs, 'b-o') if plot else None

            if abs(llh - llh_prev) < threshold:
                converged = True
                print('\nConverged after %d iterations' % i)
                break

            # M-step: Update means, covs and pie based on current responsibilities
            means, covs, pie = update_mixture_params(X, gamma)

            # Visualize the new parameters
            if plot:
                art = utils.cond_plot(it=i, title='M-step (new means)', data=means,
                      covs=covs, colors=ccolors, marker=means_marker, art=art, ax=ax1)

        if not restart and not converged:
            print('\nDid not converge after %d iterations\n' % i)

        plt.show() if plot else None

    return gamma, llhs

def main():
    file_in = 'data/old_faithful.dat'
    X = np.loadtxt(file_in, skiprows=1, usecols=(1,2))
    gamma, llhs = em_gmm(X, K=2, plot=True)

if __name__ == '__main__':
    main()
