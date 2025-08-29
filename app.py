from sklearn.model_selection import train_test_split
from diabetes_prediction.data_loader import load_data
from diabetes_prediction.model import train_model
from diabetes_prediction.evaluation import evaluate_model
from diabetes_prediction.visualization import plot_confusion_matrix
from diabetes_prediction.prediction import create_test_data

def main():
    # Load data
    df = load_data("data/pima-indians-diabetes.data.csv")

    # Split dataset
    x = df.drop('Outcome', axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(x_train, y_train)

    # Evaluate model
    accuracy, conf_matrix, class_report, y_pred = evaluate_model(model, x_test, y_test)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Predict for sample test data
    test_data = create_test_data()
    predicted_outcome = model.predict(test_data)
    print(f"Sample Prediction (Outcome): {predicted_outcome[0]}")

if __name__ == "__main__":
    main()
