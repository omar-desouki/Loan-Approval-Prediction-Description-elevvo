# Loan Approval Prediction

A machine learning project to predict loan approval status using various applicant features. This project compares different algorithms and preprocessing techniques to achieve optimal prediction performance.

## Project Overview

This project uses a loan approval dataset to build and compare machine learning models that can predict whether a loan application will be approved or rejected. The analysis includes comprehensive exploratory data analysis (EDA), data preprocessing, feature engineering, and model evaluation.

## Dataset

The dataset is sourced from Kaggle: [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

**Dataset Features:**
- `loan_id`: Unique identifier for each loan application
- `no_of_dependents`: Number of dependents of the applicant
- `education`: Education level (Graduate/Not Graduate)
- `self_employed`: Employment status (Yes/No)
- `income_annum`: Annual income of the applicant
- `loan_amount`: Requested loan amount
- `loan_term`: Loan term duration
- `cibil_score`: Credit score of the applicant
- `residential_assets_value`: Value of residential assets
- `commercial_assets_value`: Value of commercial assets
- `luxury_assets_value`: Value of luxury assets
- `bank_asset_value`: Bank asset value
- `loan_status`: Target variable (Approved/Rejected)

## Project Structure

```
├── main.ipynb              # Main Jupyter notebook with analysis
├── functions.py            # Helper functions for model training
├── data/
│   └── loan_approval_dataset.csv  # Dataset file
├── __pycache__/           # Python cache files
└── README.md              # Project documentation
```

## Analysis Pipeline

### 1. Exploratory Data Analysis (EDA)
- Data shape and basic statistics
- Distribution analysis of numerical features
- Categorical feature analysis
- Target variable distribution analysis
- Correlation analysis between features

### 2. Data Preprocessing
- **Duplicate Detection**: Checked for and handled duplicate records
- **Missing Value Analysis**: Identified and planned handling of NaN values
- **Categorical Encoding**: Applied one-hot encoding to categorical variables
- **Outlier Detection**: Used Z-score method (threshold = 3) to remove outliers
- **Feature Scaling**: Applied StandardScaler to numerical features
- **Train-Test Split**: 80-20 split with proper data leakage prevention

### 3. Feature Engineering
- **Debt-to-Income Ratio**: Created new feature combining loan amount and annual income
- **Feature Selection**: Analyzed and removed less informative features based on EDA insights

### 4. Model Development

#### Logistic Regression
- Baseline model performance
- Feature removal experiments
- Feature engineering impact
- Class balancing techniques
- **Best Performance**: 92.81% accuracy, 86.89% precision, 95.31% recall, 90.91% F1-score

#### Random Forest
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Various preprocessing combinations
- **Best Performance**: 99.41% accuracy, 100% precision, 98.44% recall, 99.21% F1-score

## Key Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression (Best) | 92.81% | 86.89% | 95.31% | 90.91% |
| Random Forest (Best) | 99.41% | 100% | 98.44% | 99.21% |

### Key Insights
1. **Random Forest significantly outperforms Logistic Regression** for this dataset
2. **Feature engineering** (debt-to-income ratio) improved model performance
3. **Class balancing** techniques were beneficial for Logistic Regression
4. **Feature removal** of less informative variables (education, self_employed) improved results
5. The dataset shows **slight class imbalance** which was addressed through balancing techniques

## Requirements

```python
# Core libraries
pandas
numpy
matplotlib
seaborn
scikit-learn

# Additional libraries
kagglehub  # For dataset download
```

## Usage

1. **Setup Environment**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```

2. **Run the Notebook**:
   - Open `main.ipynb` in Jupyter Notebook or JupyterLab
   - Execute cells sequentially
   - The notebook will automatically download the dataset if not present

3. **Model Training**:
   - Use functions from `functions.py` for model training
   - Experiment with different preprocessing techniques
   - Compare model performances

## Future Improvements

1. **Advanced Models**: Experiment with XGBoost, LightGBM, or Neural Networks
2. **Feature Engineering**: Create more domain-specific features
3. **Cross-Validation**: Implement more robust validation strategies
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Model Interpretability**: Add SHAP or LIME for model explanation

## Author

This project was developed as part of a machine learning internship task focused on loan approval prediction using various classification algorithms and preprocessing techniques.
