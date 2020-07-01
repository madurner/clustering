import matplotlib.pyplot as plt
import numpy as np
from kmeans import kmeans
from em_gmm import em_gmm
from utils import evaluate

#DOES NOT RUN YET!!!

def main():
    # Read in the dataset
    X = np.loadtxt('data/fisher_iris_data.csv', delimiter=',').T
    Y = np.loadtxt('data/fisher_iris_labels.csv', dtype=str)

    labels = list(set(Y)) # get unique labels
    num_classes = len(labels)
    Y = [labels.index(y) for y in Y]  # convert labels to integers

    ## Plot dataset
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.scatter(X[0, :], X[1, :])
    plt.title('Data without labels')
    plt.subplot(2,2,2)
    plt.scatter(X[0, :], X[1, :], c=Y)
    plt.title('Data with true labels')

    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,4)
    scatterPlot = 0
    num_evals = 100
    kmeans_performance = np.zeros(num_evals)
    kmeans_solutions = []
    for i in range(num_evals):
        print('\nK-Means Run ', i+1)
        cluster_idx = kmeans(X, num_classes, ax1=ax1, ax2=ax2)
        kmeans_performance[i] = evaluate(X, Y, cluster_idx)
        print('Accuracy: ', kmeans_performance[i])
        kmeans_solutions.append(cluster_idx)

    plt.subplot(2, 2, 3)
    plt.hist(kmeans_performance)

if __name__ == '__main__':
    main()
