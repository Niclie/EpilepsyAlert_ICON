from src.utils import constants
from scripts.run_training import run_training_logistic_regression, run_training_decision_tree, run_training_cnn
from scripts.run_preprocessing import get_dataset, get_preprocessed_dataset
from scripts.run_prediction import run_evaluation
from src.visualization.visualize import sklearn_save_learning_curve, tf_plot_all_metrics
import keras
import numpy as np


def main():
    patient_id = 'chb01'
    run(patient_id, model_type='logistic_regression')


def run(patient_id, model_type, load_from_file=True, model_path=constants.MODELS_FOLDER, plot_path=constants.PLOTS_FOLDER):
    """
    Run the whole pipeline for a given patient_id.

    Args:
    patient_id (str): patient identifier.
    model_type (str): type of model to run.
    load_from_file (bool, optional): whether to load the dataset from a file. Defaults to True.
    model_path (str, optional): path to save the cnn model. Defaults to constants.MODELS_FOLDER.
    plot_path (str, optional): path to save the plots. Defaults to constants.PLOTS_FOLDER.
    """

    match model_type:
        case 'logistic_regression':
            dataset = get_preprocessed_dataset(patient_id, load_from_file)
            if dataset is None: return None
            
            model = run_training_logistic_regression(dataset['train_data'], dataset['train_labels'])
            ev = run_evaluation(dataset['test_labels'], model.predict(dataset['test_data']), model.predict_proba(dataset['test_data']))
            sklearn_save_learning_curve(model, dataset['train_data'], dataset['train_labels'], f'{plot_path}/logistic_regression/{patient_id}', f'{patient_id}_logistic_regression')
            print(ev)
            
        case 'decision_tree':
            dataset = get_preprocessed_dataset(patient_id, load_from_file)
            if dataset is None: return None
    
            model = run_training_decision_tree(dataset['train_data'], dataset['train_labels'])
            ev = run_evaluation(dataset['test_labels'], model.predict(dataset['test_data']), model.predict_proba(dataset['test_data']))
            sklearn_save_learning_curve(model, dataset['train_data'], dataset['train_labels'], f'{plot_path}/decision_tree/{patient_id}', f'{patient_id}_decision_tree')
            print(ev)
        
        case 'cnn':
            dataset = get_dataset(patient_id, load_from_file)
            if dataset is None: return None
            
            history = run_training_cnn(dataset['train_data'], dataset['train_labels'], f'{model_path}/cnn', patient_id)
            best_epoch = np.argmin(history.history['val_loss'])
            tf_plot_all_metrics(history, best_epoch, f'{plot_path}/cnn/{patient_id}', f'{patient_id}_cnn')
            
            model = keras.models.load_model(f'{model_path}/cnn/{patient_id}.keras')
            predicted_proba = model.predict(dataset['test_data'])
            predicted_classes = (predicted_proba > 0.5).astype(int)
            ev = run_evaluation(dataset['test_labels'], predicted_classes, predicted_proba)
            print(ev)


if __name__ == '__main__':
    main()