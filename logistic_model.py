import numpy as np

def sigmoid(z):

    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):

        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def train(self, X, y):

        m, n = X.shape

        self.w = np.zeros((n,1))
        self.b = 0.0
        
        for _ in range(self.epochs):

            dec_boundary = np.dot(X, self.w) + self.b
            y_pred = sigmoid(dec_boundary)

            error = y_pred - y
            
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict_prob(self, X):
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_prob(X)

        binary_predictions = []
        for prob in probabilities:
        
            if prob[0] >= threshold:
                binary_predictions.append(1)  
            else:
                binary_predictions.append(0) 
            
    
        predictions_array = np.array(binary_predictions).reshape(-1, 1)
    
        return predictions_array






