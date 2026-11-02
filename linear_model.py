import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None  
        self.b = None  

    def train(self, X, y): # burada X matrisi, önceden elde edilen ve eğitim örneklerinin feature larını ve değerlerini içeren X_train 2D matrisidir
                           # burada y de bir sütun vektörüdür önceden bilinen gerçek değerleri kapsayan eğitim için kullanılan y_train sütun vektörüdür.
        
        m, n = X.shape # m, X 2D matrisindeki satır sayısı; n ise X matrisinin sütun sayısıdır.
        
        
        self.w = np.zeros((n, 1)) # w parametre vektörü n satırdan 1 sütundan oluşan sıfırlar matrisidir.
        self.b = 0.0 # skaler bias sayısı

        # Gradient Descent Döngüsü
        for _ in range(self.epochs): # gradyan inişini eğitim setini 1000 kere kullanarak gerçekleştir. 1000 kere w vektörü ve bias sayısı güncellenir.
           
            y_pred = np.dot(X, self.w) + self.b # (456,30) boyutlu X matrisi ile (30,1) boyutlu w matrisi çarpılır ve b eklenir. Elde edilen y_pred her örnek için sürekli bir tahmin içeren (456,1) boyutlu
            # dikey bir vektördür.

            
            error = y_pred - y  # Tahmin edilen değerlerden gerçek etiketleri (y) çıkararak hata vektörünü hesaplar: (456, 1) - (456, 1) = (456, 1).
            dw = (1 / m) * np.dot(X.T, error)  
            db = (1 / m) * np.sum(error)       

            
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict_raw(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X, threshold=0.5):
        raw_predictions = self.predict_raw(X)
    
        binary_predictions = []
    
        for pred in raw_predictions:
            if pred[0] >= threshold:
                binary_predictions.append(1)  
            else:
                binary_predictions.append(0) 
            
        predictions_array = np.array(binary_predictions).reshape(-1, 1)
    
        return predictions_array