import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Cleans data, encodes labels, and splits into train/test sets.
    """
    # 0. Check if the first row is actually a header (common in some imports)
    if df.iloc[0, 0] == 'Id' or 'Id' in df.columns:
        if df.iloc[0, 0] == 'Id':
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.reset_index(drop=True)
    
    # 1. Handle potential ID column
    df.columns = [col.lower() for col in df.columns]
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 2. Convert feature columns to numeric
    # We assume the last column is the species/target
    target_col = 'species'
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    feature_cols = [col for col in df.columns if col != target_col]
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()

    X = df[feature_cols]
    y = df[target_col]

    # 3. Encode Target Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    print(f"Data preprocessed. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, le, scaler

if __name__ == "__main__":
    pass
