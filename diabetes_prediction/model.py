# Import Necessary Libraries.
from sklearn.linear_model import LogisticRegression

# Function to Train the Logistic Regression Model.
def train_model(x_train, y_train):
    """Train and return logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model
