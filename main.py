import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from metrics import metrics 

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

evl=metrics.ClusterMetrics(X,y)
evl.elbow(X,3,1)

from sklearn.metrics.pairwise import euclidean_distances
evl.cluster_evaluation(X,y,euclidean_distances(X), max_number_of_clusters = 5)
