
ML Assignment – Multi-Model Classification with Streamlit
Objective
Develop and compare multiple supervised classification models using the Bank Marketing dataset and deploy the results through an interactive Streamlit application.
 
Dataset Overview
•	Records used: 1,000
•	Input features: 16
•	Target variable: y (binary: yes / no)
The dataset includes demographic attributes, financial indicators, and prior campaign interaction history. Both categorical and numerical features are present, requiring preprocessing and encoding.
 
Methodology
1.	Robust CSV ingestion with delimiter validation
2.	Data validation and preprocessing (encoding of categorical variables)
3.	Stratified train-test split
4.	Model training using modular architecture
5.	Performance evaluation using multiple classification metrics
 
Models Implemented
•	Logistic Regression
•	Naive Bayes
•	K-Nearest Neighbors
•	Decision Tree
•	Random Forest
•	XGBoost
Each model is implemented in a separate Python module to maintain modularity and scalability.
 
Evaluation Metrics
Given class imbalance in the target variable, relying solely on accuracy can be misleading. The following metrics were computed:
•	Accuracy
•	AUC (Area Under ROC Curve)
•	Precision
•	Recall
•	F1 Score
•	Matthews Correlation Coefficient (MCC)
AUC and MCC were emphasized as primary comparison metrics.
 
Results Summary (1,000 Rows)
With a smaller dataset (1,000 samples), model variance increased slightly due to limited training data. Observations:
•	Logistic Regression provided stable baseline performance.
•	Naive Bayes showed competitive recall but lower precision.
•	KNN performance varied depending on neighborhood sensitivity.
•	Decision Tree showed mild overfitting tendencies.
•	Random Forest improved stability over single-tree models.
•	XGBoost achieved the strongest overall balance across AUC and MCC, even with reduced data volume.
Ensemble methods demonstrated better generalization compared to linear models under reduced sample size conditions.
 
Streamlit Application Features
The deployed application enables:
•	CSV dataset upload
•	Automatic validation and preprocessing
•	Interactive model selection
•	Comparative metrics display
•	Confusion matrix visualization
The architecture supports quick experimentation across varying dataset sizes.
 
Local Execution Instructions
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

