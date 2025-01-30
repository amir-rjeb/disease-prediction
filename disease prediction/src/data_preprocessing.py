from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(file_path):
    
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data.fillna(data.mean(), inplace=True)
    
    
    data = pd.get_dummies(data, columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], drop_first=True)
    
    return data

def normalize_data(data):
    
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

def preprocess_data(file_path):
    
    data = load_data(file_path)
    data = clean_data(data)
    data = normalize_data(data)
    
    return data

def split_data(data, target_column):
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test