from sklearn.metrics import accuracy_score, confusion_matrix
import logging


def evaluate_model(model, X_test_scaled, y_test):
    try:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, cm
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise