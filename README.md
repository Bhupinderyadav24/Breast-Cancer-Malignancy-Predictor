# `# Breast Cancer Malignancy Predictor` 

### a. Problem Statement` 

```
The objective of this project is to develop a machine learning
classification system for predicting whether a breast tumor is
Benign or Malignant based on tumor characteristics.
```

```
Multiple classification models are trained and evaluated using
different performance metrics. The best-performing model is then
selected and saved for deployment.
```

```
The selected Random Forest model is deployed using Streamlit to
provide an interactive application for breast cancer malignancy
prediction.
```

```
---
```

### b. Dataset Description` 

```
The dataset used in this project is a Breast Cancer diagnostic
dataset containing numerical measurements of tumor characteristics.
```

```
The dataset contains 30 input features related to the characteristics
of cell nuclei. The features include:
```

```
- Radius
```

- `Texture` 

- `Perimeter` 

- `Area` 

- `Smoothness` 

- `Compactness` 

- `Concavity` 

- `Concave Points` 

- `Symmetry` 

- `Fractal Dimension` 

```
The features are provided in three categories:
```

`1. Mean features` 

`2. Standard Error (SE) features` 

`3. Worst features` 

```
The target variable represents the diagnosis of the tumor:
```

- `Benign` 

- `Malignant` 

```
The dataset was divided into training and testing data using an
80:20 train-test split. The training data was used to train the
classification models, while the test data was used to evaluate
their performance.
```

```
---
```

```
## c. GitHub Repository Link
```

```
GitHub Repository:
```

```
[Breast Cancer Malignancy Predictor](ADD-YOUR-GITHUB-REPOSITORY-LINK-HERE)
```

```
The GitHub repository contains the complete project implementation,
including the Streamlit application, trained model, test data,
requirements file, and project documentation.
```

```
---
```

### d. Models Used` 

```
The following machine learning classification models were
implemented and evaluated:
```

`1. Logistic Regression` 

`2. Decision Tree` 

`3. K-Nearest Neighbors (kNN)` 

`4. Gaussian Naive Bayes` 

`5. Random Forest (Ensemble)` 

#### Model Evaluation Metrics` 

```
The models were evaluated using the following metrics:
```

- `Accuracy` 

- `AUC` 

- `Precision` 

- `Recall` 

- `F1 Score` 

- `Matthews Correlation Coefficient (MCC)` 

#### Comparison Table` 

- `| ML Model | Accuracy (%) | AUC (%) | Precision (%) | Recall (%) | F1 Score (%) | MCC (%) |` 

- `|---|---:|---:|---:|---:|---:|---:|` 

- `| Logistic Regression | 96.49 | 99.60 | 97.50 | 92.86 | 95.12 | 92.45 | | Decision Tree | 92.98 | 92.46 | 90.48 | 90.48 | 90.48 | 84.92 |` 

```
| kNN | 95.61 | 98.23 | 97.44 | 90.48 | 93.83 | 90.58 |
```

- `| Gaussian Naive Bayes | 92.11 | 98.91 | 92.31 | 85.71 | 88.89 | 82.92 |` 

- `| Random Forest (Ensemble) | 97.37 | 99.29 | 100.00 | 92.86 | 96.30 | 94.42 |` 

```
---
```

### e. Observations on Model Performance` 

#### Logistic Regression` 

```
Logistic Regression performed very well on the breast cancer
dataset, achieving an accuracy of 96.49% and an AUC of 99.60%.
```

```
It also achieved a precision of 97.50%, recall of 92.86%, and
F1 score of 95.12%. These results indicate that Logistic Regression
was highly effective in distinguishing between the two classes.
```

#### Decision Tree` 

```
Decision Tree achieved an accuracy of 92.98%, which was lower than
Logistic Regression, kNN, and Random Forest.
```

```
Its precision, recall, and F1 score were all 90.48%, while its MCC
was 84.92%. Therefore, Decision Tree showed comparatively weaker
performance on this dataset.
```

#### kNN` 

```
The kNN model achieved an accuracy of 95.61% and an AUC of 98.23%.
```

```
It achieved a high precision of 97.44%, indicating that its positive
predictions were generally reliable. However, its recall of 90.48%
```

```
was lower than the recall achieved by Random Forest and Logistic
Regression.
```

```
Overall, kNN performed well but was slightly behind Random Forest
and Logistic Regression.
```

```
### Gaussian Naive Bayes
```

```
Gaussian Naive Bayes achieved an accuracy of 92.11%.
```

```
Although its AUC was relatively high at 98.91%, its recall of 85.71%
and F1 score of 88.89% were lower than the corresponding values of
the stronger models.
```

```
Therefore, Naive Bayes showed reasonable classification ability but
was not the best-performing model for this dataset.
```

```
### Random Forest (Ensemble)
```

```
Random Forest achieved the highest accuracy among the evaluated
models, with an accuracy of 97.37%.
```

```
It achieved:
```

```
- Accuracy: 97.37%
- AUC: 99.29%
- Precision: 100.00%
- Recall: 92.86%
```

```
- F1 Score: 96.30%
- MCC: 94.42%
```

```
Random Forest also achieved the highest precision, F1 score, and MCC
among the evaluated models.
```

```
Therefore, Random Forest demonstrated the strongest overall
performance on this dataset.
```

```
---
```

```
## Overall Winner
```

```
### Random Forest (Ensemble)
```

```
Random Forest was selected as the overall winning model for this
dataset.
```

```
The model achieved the highest accuracy of 97.37% and provided the
strongest overall combination of precision, recall, F1 score, and
MCC.
```

```
The trained Random Forest model was saved using Joblib as:
```

```
`model/cancer_model.pkl`
```

```
The saved model is used by the Streamlit application for making
predictions on new input data.
```

```
---
```

```
## f. Model Deployment
```

```
The selected Random Forest model is deployed using Streamlit.
```

```
The Streamlit application allows the user to enter the required
```

```
tumor characteristics and generates a prediction of whether the
tumor is Benign or Malignant.
```

```
The application uses the previously trained and saved Random Forest
model rather than retraining the model each time the application is
run.
```

```
The model is loaded from:
```

```
`model/cancer_model.pkl`
```

```
---
```

```
## g. Test Data
```

```
The project includes a `test_data.csv` file containing test-set
input features and Random Forest prediction results.
```

```
The original dataset was divided into training and testing data
using an 80:20 split. The test data was used to evaluate the trained
models.
```

```
The test data file contains the input features along with the actual
diagnosis and Random Forest prediction results.
```

```
The test data is included to demonstrate the prediction capability
of the saved Random Forest model.
```

```
---
```

```
## h. Project Structure
```

```
```text
Breast-Cancer-Malignancy-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    └── cancer_model.pkl
```

