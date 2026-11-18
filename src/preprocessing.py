"""Leakage-aware dataset validation, splitting, and preprocessing."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Excursion_Before_Delivery"
RANDOM_STATE = 42
# Exclude identifiers, date and variables plausibly derived from in-transit measurements.
LEAKAGE_COLUMNS = {
    "Shipment_ID", "Shipment_Date", "Temperature_Exposure_Risk",
    "Initial_Temperature_C", "Avg_Temperature_C", "Max_Temperature_C",
    "Min_Temperature_C", "Temperature_Std_Dev", "Temperature_Trend",
    "Temperature_Rate_of_Change", "Temperature_Threshold_Proximity",
}

def load_dataset(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "Pharmaceutical_Cold_Chain_Dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    # Keep source untouched; normalize common CSV replacement-character decoding artifacts only on read.
    return pd.read_csv(path, encoding_errors="replace")

def validate_dataset(df):
    if TARGET not in df:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    values = set(df[TARGET].dropna().unique())
    if df[TARGET].isna().any() or not values.issubset({0, 1}) or values != {0, 1}:
        raise ValueError(f"Target must contain both 0 and 1 and no missing values; found {values}")
    if df.empty:
        raise ValueError("Dataset is empty")

def split_data(df, scale=True):
    validate_dataset(df)
    y = df[TARGET].astype(int)
    excluded = LEAKAGE_COLUMNS | {TARGET}
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    # Reserve 15% test, then 15% validation from the remaining 85%.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15/.85, random_state=RANDOM_STATE, stratify=y_train)
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)], verbose_feature_names_out=False)
    Xt = prep.fit_transform(X_train)
    Xv = prep.transform(X_val)
    Xs = prep.transform(X_test)
    names = prep.get_feature_names_out()
    return {"X_train": pd.DataFrame(Xt, columns=names, index=X_train.index), "X_val": pd.DataFrame(Xv, columns=names, index=X_val.index), "X_test": pd.DataFrame(Xs, columns=names, index=X_test.index), "y_train": y_train, "y_val": y_val, "y_test": y_test, "preprocessor": prep, "feature_names": names, "raw_train": X_train, "raw_val": X_val, "raw_test": X_test, "_raw_df": df}
