# Heart Disease Prediction

## Overview
This repository contains a Jupyter Notebook that implements various machine learning models to predict heart disease using a given dataset. The notebook includes data preprocessing, exploratory data analysis, model training, evaluation, and visualization of results.

## Dataset
The dataset used for this project is assumed to contain several features related to heart disease diagnosis. Typical features may include age, sex, blood pressure, cholesterol levels, etc. The target variable is whether the patient has heart disease or not.

## Notebook Contents

### 1. Importing Libraries
The notebook starts by importing necessary libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.

### 2. Loading the Dataset
The dataset is loaded into a pandas DataFrame for analysis and preprocessing.

### 3. Data Preprocessing
- **Handling Missing Values**: Checking for and imputing or removing missing values.
- **Encoding Categorical Variables**: Transforming categorical variables into numerical representations.
- **Feature Scaling**: Normalizing or standardizing the feature values.

### 4. Exploratory Data Analysis (EDA)
- **Visualizing Distributions**: Plotting histograms and density plots.
- **Correlation Analysis**: Generating a heatmap to visualize correlations between features.
- **Pairplots**: Creating pairplots to observe relationships between features.

### 5. Model Training and Evaluation
Several models are trained and evaluated using cross-validation:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting Classifier**

Each model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

### 6. Hyperparameter Tuning
Grid search and randomized search techniques are used to find the best hyperparameters for the models, specifically for the Random Forest Classifier.

### 7. Results Visualization
- **Accuracy Scores**: Plotting the accuracy scores of different models.
- **Feature Importances**: Visualizing the importance of each feature for the Random Forest Classifier.
- **Confusion Matrix**: Displaying confusion matrices for model predictions.
- **ROC Curves**: Plotting ROC curves to compare model performances.

## Installation
To run the notebook, ensure you have the following dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd heart-disease-prediction
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heart\ Disease\ Prediction.ipynb
   ```

## Results
The notebook provides a comprehensive analysis of different machine learning models applied to the heart disease dataset. The Random Forest Classifier with tuned hyperparameters achieved the highest accuracy. Visualizations help in understanding the performance and importance of different features.

## Contributing
Feel free to contribute to this project by opening issues or submitting pull requests. Contributions are welcome to improve the code, add new models, or enhance the analysis.

## License
This project is licensed under the MIT License.

---

This README provides an overview and instructions for running and understanding the heart disease prediction notebook. It includes sections on dataset handling, preprocessing, model training, evaluation, and results visualization.
