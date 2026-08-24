import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # print(X.shape)
        # print(y.shape)
        X = np.column_stack((np.ones(np.size(X)), X))
        # y = y.reshape(1000,1)
        # print(X.shape)
        # print(y.shape)

        theta = np.linalg.pinv(X) @ y

        for i in range(self.n_iterations):
            cost = 0

            # Shuffle data
            r = np.random.randint(0, len(X))
            X = X[r]
            y = y[r]

            # SGD
            for xi, yi in zip(X, y):
                # Make prediction
                y_hat = theta @ X
                # Gradient weights and bias
                grad_w = -2*y_hat*(1-y_hat) * (yi-y_hat) * xi
                grad_b = -2*y_hat*(1-y_hat) * (yi-y_hat)
                # Update weigths and bias of model
                self.weights += -self.lr * grad_w
                self.bias += -self.lr * grad_b
                # cost-function
                cost += (y - y_hat)**2
            # Loss-function and added to history, MSE
            self.loss_history.append(cost)

            
        # ====================================
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")