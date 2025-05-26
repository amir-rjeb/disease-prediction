# Kidney Disease Prediction Project

This project aims to predict the need for dialysis in patients with kidney disease using machine learning techniques. The data used for training and prediction comes from the `data/kidney_disease.csv` file.

## Project Structure

* **data/**: Contains raw data.

  * `kidney_disease.csv`: Dataset related to kidney disease.

* **notebooks/**: Contains Jupyter notebooks for data analysis.

  * `data_analysis.ipynb`: Exploratory data analysis with visualizations and descriptive statistics.

* **src/**: Contains source code for data preprocessing, model training, evaluation, and prediction.

  * `data_preprocessing.py`: Functions for cleaning and normalizing data.
  * `model_training.py`: Code to train the prediction model.
  * `model_evaluation.py`: Evaluation of model performance.
  * `prediction.py`: Functions to make predictions using the trained model.

* **models/**: Contains the trained model.

  * `trained_model.pkl`: The saved model in pickle format.

* **requirements.txt**: List of dependencies required to run the project.

## Techniques Used

* Data Cleaning and Preprocessing
* Feature Selection and Encoding
* Classification using Machine Learning algorithms (e.g., Random Forest, Decision Tree, Logistic Regression)
* Model Evaluation using metrics such as accuracy, precision, recall, and F1-score
* Data Visualization for Exploratory Data Analysis

## Libraries Used

* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `jupyter`

## Instructions

1. **Set up the environment**:

   * Clone the repository and navigate to the project directory.
   * Install the dependencies by running:

     ```
     pip install -r requirements.txt
     ```

2. **Run the scripts**:

   * Use `data_preprocessing.py` to prepare the data.
   * Run `model_training.py` to train the model.
   * Evaluate the model using `model_evaluation.py`.
   * Use `prediction.py` to make predictions on new data.

3. **Using the model**:

   * Load the model from `models/trained_model.pkl` to make predictions about the need for dialysis.

## Authors

* \Amir Rjeb - \ amirrjb166@gmail.com
* \linkedin- \ https://www.linkedin.com/in/rjeb-amir-0866bb250/


