import numpy as np
from src.data_preprocessing.load_data import load_summary_from_file
from src.data_preprocessing.preprocess import make_dataset, preprocess_dataset
from src.utils import constants
import pandas as pd


def get_preprocessed_dataset(patient_id, load_from_file=True, out_path = constants.PREPROCESSED_FOLDER):
    """
    Get the preprocessed dataset for a given patient.

    Args:
        patient_id (str): the ID of the patient.
        load_from_file (bool, optional): whether to load the dataset from a file. Defaults to True.
        out_path (str, optional): the path to save the preprocessed dataset. Defaults to constants.PREPROCESSED_FOLDER.

    Returns:
        _type_: _description_
    """
    dataset = get_dataset(patient_id)
    dir = f'{out_path}/{patient_id}'
    
    if load_from_file:
        x_train = pd.read_csv(f'{dir}/train_data.csv')
        y_train = np.load(f'{dir}/train_labels.npy')
        
        x_test = pd.read_csv(f'{dir}/test_data.csv')
        y_test = np.load(f'{dir}/test_labels.npy')
    else:
        x_train, y_train = preprocess_dataset(dataset['train_data'], dataset['train_labels'], 256, True, dir, 'train')
        x_test, y_test = preprocess_dataset(dataset['test_data'], dataset['test_labels'], 256, True, dir, 'test')
    
    return {'train_data': x_train, 'train_labels': y_train, 'test_data': x_test, 'test_labels': y_test}


def get_dataset(patient_id, load_from_file=True, verbose=True, split=True):
    """
    Get the dataset for a given patient.

    Args:
        patient_id (str): the ID of the patient.
        load_from_file (bool, optional): whether to load the dataset from a file. Defaults to True.
        verbose (bool, optional): whether to print information about the dataset. Defaults to True.
        split (bool, optional): whether to split the dataset into training and test sets. Defaults to True

    Returns:
        dict: the dataset for the patient.
    """
    if load_from_file:
        try:
            npz = np.load(f'{constants.DATASETS_FOLDER}/{patient_id}.npz')
            data = {k: npz.get(k) for k in npz}
            npz.close()
        except:
            if verbose: print(f'Dataset for {patient_id} not found')
            return None
    else:
        patient = load_summary_from_file(patient_id)
        data = make_dataset(patient, split=split)
    
    if verbose:
        if 'train_data' in data.keys():
            print(f'Training data shape: {data['train_data'].shape}')
            print(f'Test data shape: {data['test_data'].shape}')
        else:
            print(f'Data shape: {data['data'].shape}')

    return data