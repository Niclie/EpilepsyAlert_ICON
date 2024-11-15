from src.model.predict import predict
from src.model.evaluate_model import evaluate_model


def run_evaluation(actuals, predicted_classes, predicted_proba):
    """
    Run evaluation on the model predictions.

    Args:
        actuals (np.array): array of actual values of the target variable
        predicted_classes (np.array): array of predicted classes
        predicted_proba (np.array): array of predicted probabilities

    Returns:
        tuple: a tuple containing the evaluation metrics
    """

    return evaluate_model(actuals, predicted_classes, predicted_proba)


def run_prediction(model, data):
    """
    Run prediction using the given model on the provided data.

    Args:
        model (keras.Model): the trained model object to use for prediction.
        data (numpy.ndarray): the input data to make predictions on.
    """
    
    return predict(model, data)
