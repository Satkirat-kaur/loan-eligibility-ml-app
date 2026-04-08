import os
from src.utils import setup_logging
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import encode_features, split_features_target
from src.model_training import train_model, save_artifacts
from src.model_evaluation import evaluate_model


def main():
    os.makedirs("models", exist_ok=True)
    setup_logging()

    df = load_data("data/credit.csv")
    df = preprocess_data(df)
    df = encode_features(df)

    print(df["Loan_Approved"].dtype)
    print(df["Loan_Approved"].value_counts(dropna=False))

    X, y = split_features_target(df)

    model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = train_model(X, y)

    accuracy, cm = evaluate_model(model, X_test_scaled, y_test)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    save_artifacts(model, scaler, X.columns.tolist())


if __name__ == "__main__":
    main()