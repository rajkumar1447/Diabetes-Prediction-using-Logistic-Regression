# Import Necessary Libraries.
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix heatmap."""
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
