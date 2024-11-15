from src.model.train_model import logistic_regression, decision_tree, cnn


def run_training_logistic_regression(training_data, label):
    """
    Script to run training for logistic regression

    Args:
        training_data (pd.DataFrame): Data to train the model
        label (np.array): Label for the training data

    Returns:
        model: Trained model
    """
    return logistic_regression(training_data, label)
    
    
def run_training_decision_tree(training_data, label):
    """
    Script to run training for decision tree

    Args:
        training_data (pd.DataFrame): Data to train the model
        label (np.array): Label for the training data

    Returns:
        model: Trained model
    """
    
    return decision_tree(training_data, label)


def run_training_cnn(training_data, label, out_path, file_name):
    """
    Script to run training for CNN

    Args:
        training_data (np.array): data to train the model
        label (np.array): label for the training data
        out_path (str): path to save the model
        file_name (str): name of the model file

    Returns:
        history: training history
    """
    return cnn(training_data, label, out_path, file_name)
    