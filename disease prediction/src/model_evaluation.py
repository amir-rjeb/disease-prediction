from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Évaluation du modèle:")
    print(f"Précision: {accuracy:.2f}")
    print(f"Précision (weighted): {precision:.2f}")
    print(f"Rappel (weighted): {recall:.2f}")
    print(f"AUC ROC: {roc_auc:.2f}")
    print("Matrice de confusion:")
    print(conf_matrix)

def check_for_dialysis(y_pred):
    if 1 in y_pred:
        print("Le modèle indique que certains patients ont besoin de dialyse.")
    else:
        print("Le modèle indique que aucun patient n'a besoin de dialyse.")