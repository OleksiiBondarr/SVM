import numpy as np
import numpy
import cvxopt.solvers
import logging


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVMTrainer(object):
    def __init__(self, kernel, c=0.1):
        self._kernel = kernel
        self._c = c


    def train(self, X, y):
        """
            X: martix of features
            y: vector of labels

            next step: Compute lagrange multipliers by calling _compute_lagrange_multipliers method
            retrun:    Return Predictor object by calling _create_predictor method
        """
        lagrange_multipliers = self._compute_lagrange_multipliers(X, y)
        return self._create_predictor(X, y, lagrange_multipliers)


    def _kernel_matrix(self, X):
        """
            X: martix of features

            next step: Get number of samples
            next step: Create zero matrix of quadratic shape of number of samples 
            next step: Calculate kernels
            retrun:    Return Kernels matrix
        """
        n_samples = X.shape[0]

        K = np.zeros((n_samples, n_samples))

        print(X)

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)

        return K


    def _create_predictor(self, X, y, lagrange_multipliers):
        """
            X: martix of features
            y: vector of labels
            lagrange_multipliers: vector of langrange multipliers

            next step: Get non-zero lagrange multipliers indicies
            next step: Get non-zero lagrange multipliers
            next step: Get support vecorts
            next step: Get support vecort labels
            next step: Сompute bias (use avg trick)
            retrun   : Return SVMPredictor object
        """

        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]

        support_vectors = X[support_vector_indices]

        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                    kernel=self._kernel,
                    bias=0.0,
                    weights=support_multipliers,
                    support_vectors=support_vectors,
                    support_vector_labels=support_vector_labels
                ).predict(x_k) for (y_k, x_k) in zip(support_vector_labels, support_vectors)
            ]
        )

        return SVMPredictor(
            kernel=self._kernel,
            bias=0.0,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels
        )


    def _compute_lagrange_multipliers(self, X, y):
        """
            X: martix of features
            y: vector of labels


            Need to Solve
                min 1/2 x^T P x + q^T x (aplha is x)
                s.t.
                    Gx <= h (alpha >= 0)
                    Ax = b (y^T * alpha = 0)


            next step: Get number of samples
            next step: Create Kernel matrix by calling _kernel_matrix method
            next step: Create create quadratic term P based on Kernel matrix
            next step: Create linear term q
            next step: Create G, h, A, b
            next step: Solve with - cvxopt.solvers.qp(P, q, G, h, A, b)
            retrun:    Return flatten solution['x']
        """


        n_samples = X.shape[0]

        K = self._kernel_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)

        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))

        h = cvxopt.matrix(np.zeros(n_samples))

        A = cvxopt.matrix(y, (1, n_samples))

        b = cvxopt.matrix(0.0)


        # Check this

        # -a_i \leq 0
        # G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        # h_std = cvxopt.matrix(np.zeros(n_samples))

        # # a_i \leq c
        # G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        # h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        # G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        # h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(
                self,
                kernel,
                bias,
                weights,
                support_vectors,
                support_vector_labels
            ):
        
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels


        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)


        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for w_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += w_i * y_i * self._kernel(x_i, x)

        return np.sign(result).item()
