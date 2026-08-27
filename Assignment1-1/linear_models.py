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

        m, n = np.shape(X)
        self.weights = np.random.uniform()
        self.bias = np.random.uniform()

        for _ in range(self.n_iterations):
            r = np.random.randint(m)
            xi = X[r]
            yi = y[r]

            # Make prediction
            y_hat = xi * self.weights + self.bias
            # Error
            error = y_hat - yi

            # Gradient weights and bias
            grad_w = 2 * error * xi
            grad_b = 2 * error

            # Update weigths and bias of model
            self.weights += -self.lr * grad_w
            self.bias += -self.lr * grad_b

            # Prediksjon over hele datasettet
            pred = X * self.weights + self.bias
            loss = np.mean((y - pred) ** 2)
            self.loss_history.append(loss)
        return self
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
        return X * self.weights + self.bias
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
        m, n = np.shape(X)
        self.weights = np.random.uniform(size=n)
        self.bias = np.random.uniform()

        for _ in range(self.n_iterations):
            pred = self.predict_proba(X)
            # print("gradienter")
            grad_w = (pred - y) @ X / m
            grad_b = np.mean(pred - y)

            # print("Oppdater")
            self.weights += -self.lr * grad_w
            self.bias += -self.lr * grad_b

            eps = 1e-5
            pred = np.clip(pred, eps, 1-eps)
            loss = np.mean(-y * np.log(pred) - (1 - y) * np.log(1 - pred))
            self.loss_history.append(loss)
        return self
        # raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        # print("Proba")
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
        # raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        return [1.0 if _y > 0.5 else 0.0 for _y in self.predict_proba(X)]
        # return (self.predict_proba(X) > 0.5).astype(float)
        # raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))
        # raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")