import pandas as pd
import logging

CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        # Convert target safely
        if "Loan_Approved" in df.columns:
            df["Loan_Approved"] = (
                df["Loan_Approved"]
                .astype(str)
                .str.strip()
                .replace({"Y": 1, "N": 0, "1": 1, "0": 0})
            )
            df["Loan_Approved"] = pd.to_numeric(df["Loan_Approved"], errors="coerce")

        # One-hot encode only columns that actually exist
        cols_to_encode = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)

        return df

    except Exception as e:
        logging.error(f"Encoding failed: {e}")
        raise


def split_features_target(df: pd.DataFrame):
    try:
        X = df.drop("Loan_Approved", axis=1)
        y = df["Loan_Approved"]

        # Final safety cleanup
        y = pd.to_numeric(y, errors="coerce")
        valid_idx = y.notna()

        X = X.loc[valid_idx].copy()
        y = y.loc[valid_idx].astype(int)

        return X, y

    except Exception as e:
        logging.error(f"Feature-target split failed: {e}")
        raise