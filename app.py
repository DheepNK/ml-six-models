import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import model training functions
from model.logistic_regression import train_model as train_lr
from model.decision_tree import train_model as train_dt
from model.knn import train_model as train_knn
from model.naive_bayes import train_model as train_nb
from model.random_forest import train_model as train_rf
from model.xgboost_model import train_model as train_xgb


# Streamlit Config
st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("ML Model Comparison - Bank Marketing Dataset")


# Session State Initialization
if "results" not in st.session_state:
    st.session_state.results = None

if "results_df" not in st.session_state:
    st.session_state.results_df = None

# File load
data_source = st.radio(
    "Choose how you want to load data:",
    ["Use default dataset", "Upload custom CSV"],
    index=0
)

df = None
uploaded = None
results = {}


if data_source == "Use default dataset":
    try:
        df = pd.read_csv("data/bank-full.csv", sep=";")
        st.success("Default dataset loaded.")
    except Exception:
        st.error("Default dataset not found in data/ folder.")
        st.stop()
else:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue. (Use default dataset if you don’t want to upload.)")
        st.stop()
    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";")
        if df.shape[1] == 1:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
        st.success("Uploaded dataset loaded.")
    except Exception:
        st.error("Unable to read uploaded file.")
        st.stop()

st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic Validation
if "y" not in df.columns:
    st.error("Target column 'y' not found in dataset.")
    st.stop()

if df.shape[0] < 200:
    st.warning("Dataset too small for reliable training.")
    st.stop()

# Preprocessing
df = df.copy()

# Encode target
le = LabelEncoder()
df["y"] = le.fit_transform(df["y"])

# One-hot encode categorical features
X = pd.get_dummies(df.drop(columns=["y"]), drop_first=True)
y = df["y"]

# Train-test split
test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=int(random_state),
    stratify=y
)

# Model Dictionary
models = {
    "Logistic Regression": train_lr,
    "Decision Tree": train_dt,
    "KNN": train_knn,
    "Naive Bayes": train_nb,
    "Random Forest": train_rf,
    "XGBoost": train_xgb
}

# Train Models Button
st.subheader("Train & Compare Models")

if st.button("Train models and compute metrics", type="primary"):

    results = {}

    with st.spinner("Training models..."):
        for name, train_func in models.items():
            results[name] = train_func(X_train, y_train, X_test, y_test)

    # Convert to DataFrame for comparison table
    results_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": metrics["Accuracy"],
        "AUC": metrics["AUC"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "MCC": metrics["MCC"],
    }
    for name, metrics in results.items()
    ]).sort_values(by=["MCC", "AUC"], ascending=False)

    st.session_state.results = results
    st.session_state.results_df = results_df

    st.success("Training complete.")

# Display Results
if st.session_state.results_df is not None:

    results = st.session_state.results
    results_df = st.session_state.results_df

    st.subheader("Model Comparison (sorted by MCC, then AUC)")
    st.dataframe(results_df.round(4), use_container_width=True)

    st.subheader("Detailed View")

    model_name = st.selectbox(
        "Select a model",
        results_df["Model"].tolist()
    )

    metrics = results[model_name]

    col1, col2 = st.columns(2)

    with col1:
        st.write("Metrics")
        st.json({
            k: float(v)
            for k, v in metrics.items()
            if k != "ConfusionMatrix"
        })

    with col2:
        st.write("Confusion Matrix")
        cm = metrics["ConfusionMatrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["True 0", "True 1"],
            columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_df, use_container_width=True)

    st.caption("Note: Due to class imbalance, MCC and AUC provide better performance insight than accuracy alone.")
else:
    st.info("Click 'Train models and compute metrics' to run the comparison.")

    

