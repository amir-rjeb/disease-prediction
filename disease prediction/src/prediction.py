import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import fitz  
import pandas as pd
import joblib
import os

def load_model(model_path):
    
    return joblib.load(model_path)

def preprocess_input(data):
    """Prétraiter les données d'entrée pour la prédiction."""
    column_mapping = {
        "age": "age", "blood pressure (bp)": "bp", "specific gravity (sg)": "sg", "albumin (al)": "al",
        "sugar (su)": "su", "red blood cells (rbc)": "rbc", "pus cells (pc)": "pc", "pus cell casts (pcc)": "pcc",
        "bacteria (ba)": "ba", "blood glucose random (bgr)": "bgr", "blood urea (bu)": "bu",
        "serum creatinine (sc)": "sc", "sodium (sod)": "sod", "potassium (pot)": "pot", "hemoglobin (hemo)": "hemo",
        "packed cell volume (pcv)": "pcv", "white blood cells (wc)": "wc", "red blood cells (rc)": "rc",
        "hypertension (htn)": "htn", "diabetes mellitus (dm)": "dm", "coronary artery disease (cad)": "cad",
        "appetite (appet)": "appet", "peripheral edema (pe)": "pe", "anemia (ane)": "ane"
    }
    
    data = data.rename(columns=column_mapping)
    
    data['rbc'] = data['rbc'].map({'normal': 0, 'abnormal': 1})
    data['pc'] = data['pc'].map({'normal': 0, 'abnormal': 1})
    data['pcc'] = data['pcc'].map({'not present': 0, 'present': 1})
    data['ba'] = data['ba'].map({'not present': 0, 'present': 1})
    data['htn'] = data['htn'].map({'no': 0, 'yes': 1})
    data['dm'] = data['dm'].map({'no': 0, 'yes': 1})
    data['cad'] = data['cad'].map({'no': 0, 'yes': 1})
    data['appet'] = data['appet'].map({'poor': 0, 'good': 1})
    data['pe'] = data['pe'].map({'no': 0, 'yes': 1})
    data['ane'] = data['ane'].map({'no': 0, 'yes': 1})
    
    numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data['id'] = 0  
    
    columns_order = ['id', 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    data = data[columns_order]
    
    return data

def predict_dialysis(model, input_data):
    
    return model.predict(input_data)

def extract_data_from_pdf(pdf_path):
    
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    
    data_dict = {}
    for line in text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            data_dict[key.strip().lower()] = value.strip()
    
    data = pd.DataFrame([data_dict])
    numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        try:
            input_data = extract_data_from_pdf(file_path)
            input_data = preprocess_input(input_data)
            prediction = predict_dialysis(model, input_data)
            result_text = "Oui" if prediction[0] == 1 else "Non"
            
            for i in tree.get_children():
                tree.delete(i)
            for col in input_data.columns:
                tree.insert("", "end", values=(col, input_data[col].values[0]))
            
            result_label.config(text=f"Nécessité de dialyse : {result_text}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

if __name__ == "__main__":
    model_path = 'C:/Users/Amir rjeb/Desktop/Nouveau dossier/disease prediction/models/trained_model.pkl'
    model = load_model(model_path)
    
    root = tk.Tk()
    root.title("Analyse de Dialyse")
    
    tk.Label(root, text="Importer un fichier PDF d'analyse").pack(pady=10)
    tk.Button(root, text="Ouvrir un fichier PDF", command=open_file).pack(pady=10)
    
    columns = ("Paramètre", "Valeur")
    tree = ttk.Treeview(root, columns=columns, show='headings')
    tree.heading("Paramètre", text="Paramètre")
    tree.heading("Valeur", text="Valeur")
    tree.pack(pady=10)
    
    result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
    result_label.pack(pady=10)
    
    root.mainloop()
