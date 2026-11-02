import numpy as np
from process import process_data, load_data
from linear_model import LinearRegression
from logistic_model import LogisticRegression

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def main():
    process_data(input_path='data.csv', output_path='cleaned_data.csv')
    X_train, X_test, y_train, y_test = load_data('cleaned_data.csv', test_ratio=0.2, random_seed=42)

    lin_model = LinearRegression(learning_rate=0.01, epochs=10000)
    lin_model.train(X_train, y_train)
    lin_preds = lin_model.predict(X_test, threshold=0.5)
    lin_acc = calculate_accuracy(y_test, lin_preds)

    log_model = LogisticRegression(learning_rate=0.01, epochs=10000)
    log_model.train(X_train, y_train)
    log_preds = log_model.predict(X_test, threshold=0.5)
    log_acc = calculate_accuracy(y_test, log_preds)

    print(f"Lineer Regresyon Başarısı  : %{lin_acc:.2f}")
    print(f"Lojistik Regresyon Başarısı: %{log_acc:.2f}")

if __name__ == "__main__":
    main()