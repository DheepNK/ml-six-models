import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="ML Assignment - Model Comparison", layout="wide")

st.title("ML Assignment 2 – Model Comparison (Streamlit)")
st.write("Upload a dataset (Bank Marketing) or use the bundled sample, click the button Train models, and compare metrics.")

# -------------------------
# Helpers
# -------------------------
def build_preprocessor(numeric_features, categorical_features, dense_for_nb=False):
    if dense_for_nb:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_encoder = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", cat_encoder, categorical_features),
        ]
    )

def evaluate_binary(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Some classifiers may not support predict_proba; here all chosen ones do.
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_prob)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred),
    }

@st.cache_data
def load_default_data():
    # Uses the file you already downloaded
    return pd.read_csv("data/bank-full.csv", sep=";")

@st.cache_resource
def train_models(df, test_size=0.2, random_state=42):
    # Robust target mapping
    df = df.copy()
    df["y"] = df["y"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=["y"])

    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    numeric_features = [
        "age", "balance", "day", "duration",
        "campaign", "pdays", "previous"
    ]
    categorical_features = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Shared preprocessor (sparse OK)
    preprocessor = build_preprocessor(numeric_features, categorical_features, dense_for_nb=False)
    # NB needs dense
    nb_preprocessor = build_preprocessor(numeric_features, categorical_features, dense_for_nb=True)

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42))
        ]),
        "KNN (k=5)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes (Gaussian)": Pipeline([
            ("preprocessor", nb_preprocessor),
            ("classifier", GaussianNB())
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                n_jobs=-1
            ))
        ]),
    }

    # Train and evaluate
    results = {}
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        metrics = evaluate_binary(pipe, X_test, y_test)
        results[name] = metrics
        trained[name] = pipe

    # Build comparison table
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"],
        })
    results_df = pd.DataFrame(rows).sort_values(by=["MCC", "AUC"], ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    return trained, results, results_df

# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("Controls")
    source = st.radio("Dataset source", ["Use bundled dataset", "Upload CSV"], index=0)

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)

    st.caption("Note: Bank Marketing CSV is semicolon-separated. Target column must be `y` with yes/no.")

# Load data
if source == "Use bundled dataset":
    df = load_default_data()
    
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    print(uploaded)
    if uploaded is None:
        st.info("Please upload a CSV file to proceed.")
        st.stop()

    # Try common delimiters
    try:
        df = pd.read_csv(uploaded, sep=";")
        if df.shape[1] == 1:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)  # fallback comma
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or unreadable.")
        st.stop()

    except Exception as e:
        st.error("Failed to read the uploaded CSV file.")
        st.exception(e)
        st.stop()
# Show dataset summary
st.subheader("Dataset Preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows", df.shape[0])
with c2:
    st.metric("Columns", df.shape[1])
with c3:
    st.write("Columns:", ", ".join(list(df.columns)[:10]) + (" ..." if len(df.columns) > 10 else ""))

st.dataframe(df.head(20), use_container_width=True)

if "y" not in df.columns:
    st.error("Target column `y` not found. Please upload the Bank Marketing dataset (or ensure your target column is named `y`).")
    st.stop()

# Train models
st.subheader("Train & Compare Models")
if st.button("Train models and compute metrics", type="primary"):
    with st.spinner("Training models..."):
        trained, results, results_df = train_models(df, test_size=test_size, random_state=int(random_state))

    st.success("Done.")

    st.subheader("Model Comparison (sorted by MCC, then AUC)")
    st.dataframe(results_df.round(4), use_container_width=True)

    # Pick model for detailed view
    st.subheader("Detailed View")
    model_name = st.selectbox("Select a model", results_df["Model"].tolist(), index=0)
    m = results[model_name]

    left, right = st.columns([1, 1])
    with left:
        st.write("Metrics")
        st.json({k: v for k, v in m.items() if k != "ConfusionMatrix"})
    with right:
        st.write("Confusion Matrix (rows: true, cols: predicted)")
        cm = m["ConfusionMatrix"]
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

    st.caption("Tip: Accuracy can look high due to class imbalance. MCC and AUC are better overall indicators here.")
else:
    st.info("Click **Train models and compute metrics** to run the full comparison.")

