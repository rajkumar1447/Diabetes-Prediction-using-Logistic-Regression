# Import Necessary Libraries.
import pandas as pd

# Function to Test Data Preperation.S
def create_test_data():
    """Return a sample test data DataFrame for prediction."""
    test_data = pd.DataFrame({
        'Pregnancies': [6],
        'Glucose': [148],
        'BloodPressure': [72],
        'SkinThickness': [35],
        'Insulin': [0],
        'BMI': [33.6],
        'DiabetesPredigreeFunction': [0.627],
        'Age': [50]
    })
    return test_data
