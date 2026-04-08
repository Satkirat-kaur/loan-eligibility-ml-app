import pandas as pd
import logging


def load_data(path: str) -> pd.DataFrame:
    try:
        logging.info(f"Loading dataset from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        # Convert columns to object type as done in notebook
        df["Credit_History"] = df["Credit_History"].astype("object")
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

        # Fill missing values
        df["Gender"] = df["Gender"].fillna("Male")
        df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
        df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
        df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
        df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

        # Drop Loan_ID
        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)

        return df

    except Exception as e:
        logging.error(f"Failed during preprocessing: {e}")
        raise