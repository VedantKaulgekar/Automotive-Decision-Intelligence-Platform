# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_dataset(uploaded_file):
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "unknown"
        raise ValueError(
            f"Unsupported file format: .{ext}\n\n"
            "Only CSV (.csv) and Excel (.xlsx, .xls) files are accepted for model training.\n"
            "If your data is in a PDF or another format, please export it to CSV or Excel first, "
            "then re-upload."
        )


def preprocess(df):
    df = df.copy()

    # Keep non-numeric columns (targets!)
    non_numeric = df.select_dtypes(exclude=["float64", "int64"]).copy()

    # Only scale numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64"]).copy()

    # Impute numeric missing values
    imputer = SimpleImputer(strategy="median")
    cleaned = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

    # Scale numeric
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(cleaned), columns=numeric_df.columns)

    # Reattach non-numeric columns
    cleaned_full = pd.concat([cleaned, non_numeric], axis=1)

    return cleaned_full, scaled, imputer, scaler