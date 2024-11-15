from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def evaluate_model(actuals, predicted_classes, predicted_proba):
    """
    Returns the following metrics given the actuals, predicted classes and predicted probabilities:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    - Log Loss

    Args:
        actuals (np.array): array of actual values of the target variable
        predicted_classes (np.array): array of predicted classes
        predicted_proba (np.array): array of predicted probabilities

    Returns:
        tuple: a tuple containing the aforementioned metrics
    """
    
    accuracy = accuracy_score(actuals, predicted_classes)
    precision = precision_score(actuals, predicted_classes)
    recall = recall_score(actuals, predicted_classes)
    f1 = f1_score(actuals, predicted_classes)
    cm = confusion_matrix(actuals, predicted_classes)
    logloss = log_loss(actuals, predicted_proba)
    
    p_naive = actuals.mean()
    naive_log_loss = log_loss(actuals, np.full_like(actuals, p_naive))
    
    return (accuracy, precision, recall, f1, cm, logloss, naive_log_loss)