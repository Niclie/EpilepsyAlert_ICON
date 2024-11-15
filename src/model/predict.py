def predict(model, data):
    """
    Predict the data using the model.

    Args:
        model (keras.Model): trained model used for prediction.
        data (array-like): input data to be predicted.

    Returns:
        list: a list of predicted probabilities for each input data point.
    """

    predictions = model.predict(data)

    return [probs for probs in predictions]