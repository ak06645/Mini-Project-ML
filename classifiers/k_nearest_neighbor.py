import numpy as np

class KNearestNeighbor():
    """ a kNN classifier with L2 distance 
        Our conventions:
            N: Number of trainig samples
            D: Dimensionality (Number of features)
    """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier.
        For k-nearest neighbors this is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (N, D) containing the training data
          consisting of N samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for query data using this classifier.

        Inputs:
        - X: A numpy array of shape (Nq, D) containing query data consisting
             of Nq samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and query points.

        Returns:
        - y: A numpy array of shape (Nq,) containing predicted labels for the
          query data, where y[i] is the predicted label for the query sample X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each query point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_query = X.shape[0]  #Number of query samples: Nq
        num_train = self.X_train.shape[0]#Number of training samples: N
        dists = np.zeros((num_query, num_train))

        def dist(X, Y):
            sx = np.sum(X**2, axis=1, keepdims=True)
            sy = np.sum(Y**2, axis=1, keepdims=True)
            return np.sqrt(-2 * X.dot(Y.T) + sx + sy.T)

        dists = dist(X, self.X_train)
    
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between query points and training points,
        predict a label for each query point.

        Inputs:
        - dists: A numpy array of shape (num_query, num_train) where dists[i, j]
          gives the distance betwen the ith query point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_query,) containing predicted labels for the
          query data, where y[i] is the predicted label for the query point X[i].
        """
        num_query = dists.shape[0]#Number of query samples: Nq
        y_pred = np.zeros(num_query)
        for i in range(num_query):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith query point.
            closest_y = []

            near = np.argsort(dists[i])[:k]
            for j in range(0,k) :
                closest_y.append( self.y_train[ near[j] ] )

            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
