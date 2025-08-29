from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report, y_pred
