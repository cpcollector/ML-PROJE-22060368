import pandas as pd
import numpy as np

def process_data(input_path='data.csv', output_path='cleaned_data.csv'):
    df = pd.read_csv(input_path)
    
    cols_to_drop = ['id', 'Unnamed: 32']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    df.to_csv(output_path, index=False)
    print(f"[BAŞARILI] Temizlenmiş veri '{output_path}' dosyasına yazıldı.")

def load_data(cleaned_path='cleaned_data.csv', test_ratio=0.2, random_seed=42):
    df = pd.read_csv(cleaned_path)
    
    X = df.drop(columns=['diagnosis']).values
    y = df['diagnosis'].values.reshape(-1, 1)
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    indices = np.random.permutation(X.shape[0])
    test_size = int(test_ratio * len(X))
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    X_train_raw, X_test_raw = X[train_idx], X[test_idx] 
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_min = np.min(X_train_raw, axis=0)
    X_max = np.max(X_train_raw, axis=0)
    
    X_train = (X_train_raw - X_min) / (X_max - X_min)
    X_test = (X_test_raw - X_min) / (X_max - X_min)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    process_data('data.csv')
    X_train, X_test, y_train, y_test = load_data('cleaned_data.csv')
    
    print("--- Veri Ön İşleme Tamamlandı (Veri Sızıntısı Önlemli) ---")
    print(f"X_train Boyutu : {X_train.shape}")
    print(f"X_test Boyutu  : {X_test.shape}")
    print(f"y_train Boyutu : {y_train.shape}")
    print(f"y_test Boyutu  : {y_test.shape}")
    print(f"X_train Min    : {X_train.min():.4f} | Max: {X_train.max():.4f}")
    print(f"X_test Min     : {X_test.min():.4f} | Max: {X_test.max():.4f}")