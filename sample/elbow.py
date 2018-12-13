def elbow(data, max_number_of_clusters, step = 1):
    """Plots the elbow containing the variance of each cluster
    Args:
        data: data to be clustered
        max_number_of_clusters: maximum number of clusters
        step: determines how much the iteraction will increase (1 by default)
            For example, if step = 10, the function will plot the elbow for every 10 clusters
    Returns:
        void
    """
    data = data[0]
    distortions = []
    K = np.arange(1, max_number_of_clusters+1,step)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(data)
        kmeans.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / len(data))

    # Plot the elbow
    plt.plot(K, distortions, 'x-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
