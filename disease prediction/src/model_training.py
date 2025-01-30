import os
import zipfile
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


os.system('kaggle datasets download -d uciml/pima-indians-diabetes-database')


if not os.path.exists('pima-indians-diabetes-database.zip'):
    raise FileNotFoundError("Fichier non trouvé. Vérifiez le téléchargement.")


with zipfile.ZipFile('pima-indians-diabetes-database.zip', 'r') as zip_ref:
    zip_ref.extractall('data')


kaggle_data = pd.read_csv('data/diabetes.csv')


kaggle_data['classification'] = 'notckd'


kaggle_data = kaggle_data.rename(columns={
    'Pregnancies': 'id',
    'Glucose': 'bgr',
    'BloodPressure': 'bp',
    'SkinThickness': 'sg',
    'Insulin': 'al',
    'BMI': 'su',
    'DiabetesPedigreeFunction': 'sc',
    'Age': 'age'
})

default_values = {
    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
    'sod': 140, 'pot': 4.5, 'hemo': 14.0, 'pcv': 45, 'wc': 8000, 'rc': 5.0,
    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
}
for col, val in default_values.items():
    kaggle_data[col] = val

data = pd.read_csv('C:/Users/Amir rjeb/Desktop/Nouveau dossier/disease prediction/data/kidney_disease.csv')


data = pd.concat([data, kaggle_data], ignore_index=True)

print("Répartition des classes après ajout des exemples négatifs :")
print(data['classification'].value_counts())


data['pcv'] = pd.to_numeric(data['pcv'], errors='coerce')
data['wc'] = pd.to_numeric(data['wc'], errors='coerce')
data['rc'] = pd.to_numeric(data['rc'], errors='coerce')


numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())


categorical_mappings = {
    'rbc': {'normal': 0, 'abnormal': 1},
    'pc': {'normal': 0, 'abnormal': 1},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba': {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm': {'no': 0, 'yes': 1},
    'cad': {'no': 0, 'yes': 1},
    'appet': {'poor': 0, 'good': 1},
    'pe': {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1}
}

for col, mapping in categorical_mappings.items():
    if col in data.columns:
        data[col] = data[col].map(mapping)

categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

print("Valeurs manquantes après prétraitement :")
print(data.isnull().sum())


X = data.drop(['classification', 'Outcome'], axis=1, errors='ignore')
y = data['classification'].map({'ckd': 1, 'notckd': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Valeurs manquantes dans X_train avant SMOTE :")
print(X_train.isnull().sum())


categorical_cols_train = X_train.select_dtypes(include=['object']).columns
if not categorical_cols_train.empty:
    X_train[categorical_cols_train] = X_train[categorical_cols_train].fillna(X_train[categorical_cols_train].mode().iloc[0])


numeric_cols_train = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numeric_cols_train] = X_train[numeric_cols_train].fillna(X_train[numeric_cols_train].mean())

print("Valeurs manquantes dans X_train après remplissage :")
print(X_train.isnull().sum())

print("Types de données avant SMOTE :")
print(X_train.dtypes)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

model_dir = 'C:/Users/Amir rjeb/Desktop/Nouveau dossier/disease prediction/models'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'trained_model.pkl'))