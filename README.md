# ML Assignment – Model Comparison with Streamlit

## Problem Statement
Build and compare multiple classification models on the Bank Marketing dataset and present results through an interactive Streamlit application.

## Dataset
- Records: ~45,000
- Features: 16 input features
- Target: `y` (binary: yes / no)

The dataset contains numerical and categorical variables related to customer demographics and marketing campaign history.

## Approach
1. Data ingestion with delimiter handling
2. Basic validation and preprocessing
3. Train-test split
4. Model training and evaluation

### Models Implemented
- Logistic Regression
- Naive Bayes
- KNN
- Decision Tree
- Random Forest
- XGBoost

## Evaluation Metrics
Due to class imbalance, accuracy alone is insufficient.  
The following metrics were used:

- Accuracy  
- AUC  
- Precision  
- Recall  
- F1 Score  
- MCC (Matthews Correlation Coefficient)

AUC and MCC were prioritized for model comparison.

## Results Summary
Tree-based ensemble models outperformed linear models.  
XGBoost achieved the best overall balance across AUC and MCC.

## Streamlit Application
The app allows:
- CSV upload
- Automatic validation
- Model selection
- Metric comparison
- Confusion matrix visualization

### Run Instructions
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
