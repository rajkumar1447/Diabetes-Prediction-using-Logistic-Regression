import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load and return the diabetes dataset."""
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPredigreeFunction', 'Age', 'Outcome'
    ]
    data = pd.read_csv(file_path, names=column_names, header=None)
    return data
