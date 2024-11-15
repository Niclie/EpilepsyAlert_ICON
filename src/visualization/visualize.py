import matplotlib.pyplot as plt
from src.utils import constants
import os.path
from datetime import date, datetime
from src.utils.debug_utils import check_folder
from sklearn.metrics import log_loss
import numpy as np
from sklearn.model_selection import learning_curve


def log_metrics(patient_id, model, n_training, n_test, accuracy, loss, file_path = f'{constants.RESULTS_FOLDER}/results.csv'):
    """
    Log the metrics of the model in a csv file.

    Args:
        patient_id (str): patient id.
        n_training (int): number of training examples.
        n_test (int): number of test examples.
        accuracy (float): accuracy of the model.
        loss (float): loss of the model.
        file_path (str, optional): path to the csv file. Defaults to f'{constants.RESULTS_FOLDER}/results.csv'.
    """
    if not os.path.isfile(file_path):
        f = open(file_path, 'w')
        f.write('ID, Model, Training examples, Test examples, Accuracy, Loss, Date, Time\n')
    else:
        f = open(file_path, 'a')
        
    day = date.today().strftime('%d/%m/%Y')
    time = datetime.now().strftime('%H:%M:%S')
    f.write(f'{patient_id}, {model}, {n_training}, {n_test}, {accuracy}, {loss}, {day}, {time}\n')

    f.close()


def visualize_data(data, label, classes):
    """
    Visualizes the data for each class.

    Args:
        data (numpy.ndarray): input data.
        label (numpy.ndarray): labels corresponding to the data.
        classes (list): list of classes.
    """
    plt.figure()
    for c in classes:
        class_data = data[label == c]
        plt.plot(class_data[0].T[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()


def sklearn_save_learning_curve(model, x_train, y_train, path, file_name):
    """
    Save the learning curve of the model in the specified path with the specified file name.

    Args:
        model (sklearn model): model to be evaluated.
        x_train (numpy.ndarray): training data.
        y_train (numpy.ndarray): training labels.
        path (str): path to save the learning curve.
        file_name (str): name of the file to save the learning curve.
    """
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_train, y_train, cv=10, scoring=__log_loss_scorer, shuffle=True
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label="Training Log-Loss")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation Log-Loss")

    plt.legend(loc="best")
    plt.savefig(f'{path}/{file_name}')
    
    
def __log_loss_scorer(estimator, x, y):
    """
    Custom scorer for the learning curve function. It calculates the log loss of the model.
    """
    y_pred_proba = estimator.predict_proba(x)
    
    return log_loss(y, y_pred_proba)


def tf_plot_all_metrics(history, best_epoch, path, file_name):
    """
    Plot all metrics from the given history object and save the plots with the specified file name.

    Args:
        history (dictionary): history object containing the training metrics.
        file_name (str): name of the file to save the plots.
    """

    keys = list(history.history.keys())
    keys = keys[:len(keys)//2]
    for metric in keys:
        __tf_plot_metric(history, best_epoch, metric, path, f'{file_name}_{metric}')


def __tf_plot_metric(history, best_epoch, metric, path, file_name):
    """
    Plot the specified metric from the given history object and save the plot with the specified file name.

    Args:
        history (dictionary): history object containing the training metrics.
        metric (str): metric to be plotted.
        file_name (str): name of the file to save the plot.
    """
    check_folder(path)
    
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Saved Model (Epoch {best_epoch})')
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    
    plt.savefig(f'{path}/{file_name}')
    plt.close()