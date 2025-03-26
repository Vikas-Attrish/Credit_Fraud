# Credit Card Fraud Detection

This project implements a comprehensive credit card fraud detection system using multiple machine learning models. It aims to accurately identify fraudulent transactions by comparing the performance of different algorithms through a variety of evaluation metrics.

## 📊 Project Overview

Credit card fraud detection is a critical challenge faced by financial institutions. The primary objective is to differentiate between legitimate ("Normal") and fraudulent ("Fraud") transactions. Given the highly imbalanced nature of the dataset (where fraudulent cases are rare), the project emphasizes handling class imbalance while ensuring robust model evaluation.

This project:
- Utilizes a real-world dataset of credit card transactions.
- Applies five distinct machine learning models to detect fraudulent transactions.
- Evaluates model performance using multiple metrics (accuracy, precision, recall, F1-score).
- Visualizes the performance through confusion matrices and comparative bar charts.

## 📁 Dataset Information

The dataset is sourced from the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and contains transactions made by European cardholders in September 2013.

**Dataset Details:**
- **Total Records:** 284,807
- **Fraudulent Transactions:** 492 (0.172% of the dataset)
- **Features:** 30 columns (28 anonymized principal component analysis (PCA) features, and two additional columns: 'Time' and 'Amount')
- **Target Variable:** 'Class' (0 indicates a legitimate transaction, 1 indicates a fraudulent transaction)

> **Note:** Due to the large size of the dataset, the CSV file is not included in the repository. You can download it from the provided Kaggle link.

## 📊 Machine Learning Models Implemented

1. **Logistic Regression:**
   - A linear model used for binary classification.
   - Strengths: Simple, interpretable, and fast.
   - Weaknesses: Limited ability to capture complex patterns.

2. **Random Forest:**
   - An ensemble of decision trees that improves accuracy by averaging predictions.
   - Strengths: Robust to overfitting, handles non-linearity well.
   - Weaknesses: Computationally intensive, less interpretable.

3. **Decision Tree:**
   - A tree-based model that splits data based on feature thresholds.
   - Strengths: Easy to interpret and visualize.
   - Weaknesses: Prone to overfitting without pruning.

4. **Support Vector Machine (SVM):**
   - A model that finds the optimal hyperplane to separate classes.
   - Strengths: Effective in high-dimensional spaces.
   - Weaknesses: Computationally expensive for large datasets.

5. **Ensemble Technique (Voting Classifier):**
   - Combines multiple models (Random Forest and Logistic Regression) to make final predictions.
   - Strengths: Improved accuracy by leveraging diverse models.
   - Weaknesses: Increased complexity and training time.

## 📊 Model Performance Metrics

We evaluate the models using the following metrics:

1. **Accuracy:** Proportion of correctly classified instances.
2. **Precision:** Ratio of true positive predictions to all positive predictions.
3. **Recall (Sensitivity):** Ratio of true positive predictions to all actual positives.
4. **F1-Score:** Harmonic mean of precision and recall (useful for imbalanced datasets).

## 🛠️ Project Setup

### Prerequisites

Ensure you have Python and the following libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Directory Structure

```
Credit_Fraud/
├── creditfraud.ipynb      # Logistic Regression, Random Forest, Decision Tree, SVM
├── rfcreditfraud.ipynb    # Random Forest and Ensemble Technique
└── README.md
```

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/Vikas-Attrish/Credit_Fraud.git
cd Credit_Fraud
```

2. Download the dataset (`creditcard.csv`) from the [Kaggle dataset link](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

3. Execute the notebooks:

- **creditfraud.ipynb**: Contains the implementation and evaluation of individual models (Logistic Regression, Random Forest, Decision Tree, SVM).
- **rfcreditfraud.ipynb**: Focuses on Random Forest and Ensemble techniques, with detailed performance analysis.

## 📊 Model Performance Comparison

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.99     | 0.91      | 0.84   | 0.87     |
| Random Forest        | 0.99     | 0.97      | 0.76   | 0.85     |
| Decision Tree        | 0.99     | 0.69      | 0.79   | 0.74     |
| Support Vector Machine | 0.99   | 0.91      | 0.84   | 0.87     |
| Ensemble Technique   | 0.99     | 0.87      | 0.57   | 0.69     |

## 📈 Visualization

The project includes the following visual outputs:

1. **Confusion Matrix:**
   - Displays true vs. predicted values for each model.
2. **Performance Comparison Chart:**
   - Side-by-side comparison of accuracy, precision, recall, and F1-score for all models.

## 📌 Key Functions Explained

1. **Data Preprocessing:**
   - Load and clean the dataset.
   - Split data into training and testing sets.

2. **Model Training and Prediction:**
   - Fit each model on the training data.
   - Make predictions on the test data.

3. **Evaluation Metrics:**
   - Compute accuracy, precision, recall, and F1-score.

4. **Visualization:**
   - Generate confusion matrices and comparison charts.

## 📚 Future Improvements

1. **Advanced Ensemble Methods:**
   - Implement Gradient Boosting (XGBoost, LightGBM, CatBoost).

2. **Hyperparameter Tuning:**
   - Use Grid Search or Random Search for model optimization.

3. **Data Imbalance Handling:**
   - Apply SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.

4. **Feature Engineering:**
   - Derive new features from existing data to improve model accuracy.

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and share it.

## 🤝 Contribution Guidelines

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

## 📧 Contact Information

For questions or collaboration, contact: attrishviikas26@gmail.com

## 📝 Acknowledgements

- The dataset is provided by the Machine Learning Group at ULB.
- Inspired by the need to combat financial fraud using cutting-edge machine learning techniques.

