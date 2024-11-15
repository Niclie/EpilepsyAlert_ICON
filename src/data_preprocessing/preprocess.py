from datetime import timedelta
from src.utils import constants
import numpy as np
import time
import tsfel


def preprocess_dataset(data, labels, fs, save=True, out_path=None, file_name=None):
    """
    Preprocess the dataset with the time series features extractor.
    The features are extracted from the data and saved in a csv file, and the labels are saved in a npy file.
    The features includes kurtois, mean, root mean square, skewness, standard deviation.

    Args:
        data (np.array): data to extract the features.
        labels (np.array): labels of the data.
        fs (int): sampling frequency.
        save (bool, optional): whether to save the features and labels. Defaults to True.
        out_path (str, optional): path to save the files. Defaults to None.
        file_name (str, optional): name of the files to save. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with the extracted features and np.array with the labels.
    """
    cfg = tsfel.get_features_by_domain(json_path='src/utils/features.json')
    x = tsfel.time_series_features_extractor(cfg, data, fs=fs)
    if save and out_path is not None:
        x.to_csv(f'{out_path}/{file_name}_data.csv', index=False)
        np.save(f'{out_path}/{file_name}_labels', labels)
    
    return x, labels


def make_dataset(patient, 
                 in_path = constants.DATA_FOLDER,
                 interictal_hour = 4,
                 preictal_hour = 1,
                 segment_size = 5, 
                 balance = True, 
                 split = True, 
                 save = True, 
                 out_path = constants.DATASETS_FOLDER):
    """
    Create a dataset with the interictal and preictal states of the seizures of the patient.

    Args:
        patient (Patient): patient to create the dataset.
        in_path (str, optional): path where the recordings are stored. Defaults to constants.DATA_FOLDER.
        interictal_hour (int, optional): hours before the preictal state to consider as interictal. Defaults to 4.
        preictal_hour (int, optional): hours before the seizure to consider as preictal. Defaults to 1.
        segment_size (int, optional): size of the segments in seconds. Defaults to 5.
        balance (bool, optional): whether to balance the data. Defaults to True.
        split (bool, optional): whether to split the data into training and test sets. Defaults to True.
        save (bool, optional): whether to save the dataset on a npz file. Defaults to True.
        out_path (str, optional): path where to save the dataset. Defaults to constants.DATASETS_FOLDER.

    Returns:
        dict: dictionary with the dataset.
    """

    if in_path == constants.DATA_FOLDER:
        in_path += f'/{patient.id}'
    
    #interictal_hour = int(interictal_hour)
    #preictal_hour = int(preictal_hour)

    if segment_size < 1:
        print(f'No dataset created for {patient.id}, segment_size should be >= 1\n')
        return None
    
    grouped_recordings = min_channel_recordings(patient, 23)
    
    sampling_rate = grouped_recordings[0][0].sampling_rate
    n_interictal = int(((interictal_hour * 3600) * sampling_rate) / (segment_size * sampling_rate))

    print(f'Creating dataset for {patient.id}...')

    data = []
    for recordings in grouped_recordings:
        prev_seizure_end = recordings[0].start
        for rec in recordings:
            for s in rec.get_seizures_datetimes():
                start_preictal, end_preictal = get_phase_datetimes(recordings, s[0], preictal_hour, mod = 'static', gap = 1)
                start_interictal, end_interictal = get_phase_datetimes(recordings, start_preictal, interictal_hour, mod = 'dynamic', gap = 0)

                if not(start_interictal <= prev_seizure_end <= end_preictal):
                    temp_data = retrive_data(recordings, in_path, start_interictal, end_preictal, n_channels=23)
                    temp_data = segment_data(temp_data, segment_size, sampling_rate)
                    
                    labels = (np.zeros(n_interictal), np.ones(len(temp_data) - n_interictal))

                    data.append((temp_data, labels))

                prev_seizure_end = s[1]

    if len(data) < 1:
        print(f'No dataset created for {patient.id}\n')
        return None

    if balance:
        data = balance_data(data)
    
    if split:
        train_data, train_labels, test_data, test_labels = split_data(data)

        data = {'train_data': train_data, 
                'train_labels': train_labels, 
                'test_data': test_data, 
                'test_labels': test_labels, 
                'channels': np.array(recordings[0].channels)}
    else:
        labels = [np.append(d[1][0], d[1][1]) for d in data]
        data = {'data': np.vstack([d[0] for d in data]),
                'labels': np.hstack((labels)),
                'channels': np.array(recordings[0].channels)}
    
    if save:
        start_time = time.time()
        print(f'Creating file for {patient.id}...')

        filename = f'{out_path}/{patient.id}.npz'
        np.savez(filename, **data)
        
        print(f'File created in {time.time() - start_time:.2f} seconds\n')

    return data


def min_channel_recordings(patient, n_channels):
    """
    Filter the recordings with the minimum number of channels.

    Args:
        patient (Patient): patient to filter the recordings.
        n_channels (int): minimum number of channels.

    Returns:
        list: list of recordings with the minimum number of channels.
    """
    recordings = patient.group_by_channels(keep_order = True)
    cond = (False, 0)
    filtered_recordings = []
    for i, recs in enumerate(recordings):
        if len(recs[0].channels) >= n_channels:
            if cond[0]:
                filtered_recordings[cond[1]].extend(recs)
            else:
                filtered_recordings.append(recs)
                cond = (True, i)
        else:
            cond = (False, 0)
            
    return filtered_recordings


def split_data(data, train_size = 80, test_size = 20):
    """
    Splits the given data into training and testing sets based on the specified train size and test size.
    
    Args:
        data: the input data to be split.
        train_size (optional): the percentage of data to be used for training. Default is 80.
        test_size (optional): the percentage of data to be used for testing. Default is 20.

    Returns:
        tuple: a tuple containing:
            train_data: the training data.
            train_labels: the labels corresponding to the training data.
            test_data: the testing data.
            test_labels: the labels corresponding to the testing data.
    """

    if train_size + test_size != 100: return

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    rng = np.random.default_rng()
    for d in data:
        n_training = (len(d[0]) * train_size) // 100

        interictal_train_indices, interictal_test_indices = split_indices_randomly(np.arange(len(d[1][0])), n_training // 2, rng)
        preictal_train_indices, preictal_test_indices = split_indices_randomly(np.arange(len(d[1][0]), len(d[1][0]) + len(d[1][1])), n_training // 2, rng)

        train_data.append(
            np.vstack(
                (np.array([d[0][i] for i in interictal_train_indices]), 
                 np.array([d[0][i] for i in preictal_train_indices]))
            )
        )
        
        train_labels.append(
            np.append(
                np.zeros(len(interictal_train_indices)),
                np.ones(len(preictal_train_indices))
            )
        )

        test_data.append(
            np.vstack(
                (np.array([d[0][i] for i in interictal_test_indices]), 
                 np.array([d[0][i] for i in preictal_test_indices]))
            )
        )

        test_labels.append(
            np.append(
                np.zeros(len(interictal_test_indices)),
                np.ones(len(preictal_test_indices))
            )
        )

    return (np.vstack((train_data)), np.hstack((train_labels)), np.vstack((test_data)), np.hstack((test_labels)))


def split_indices_randomly(array, size, rng):
    """
    Split the array into two subarrays, the first one with the specified size and the second one with the remaining elements.

    Args:
        array (np.array): array to split.
        size (int): size of the first subarray.
        rng (np.random.Generator): random number generator.

    Returns:
        tuple: tuple with the first subarray and the second subarray.
    """

    first_sub = rng.choice(array, size, replace = False)
    second_sub = np.setdiff1d(array, first_sub)

    return first_sub, second_sub


def balance_data(data):
    """
    Balance the data with the same number of interictal and preictal segments.

    Args:
        data (list): list of tuples with the interictal and preictal data.

    Returns:
        list: list of tuples with the balanced data.
    """

    balanced_data = []
    rng = np.random.default_rng()
    for d in data:
        random_interictal_index = rng.choice(len(d[1][0]), size = len(d[1][1]), replace = False)
        random_interictal = np.array([d[0][i] for i in random_interictal_index])
        
        balanced_data.append((np.append(random_interictal, d[0][len(d[1][0]):], axis=0),
                              (d[1][0][:len(random_interictal_index)],
                               d[1][1])))

    return balanced_data


def segment_data(data, segment_size, sampling_rate):
    """
    Segment the data in segments of the specified size. 

    Args:
        data (np.array): data to segment.
        segment_size_seconds (int): size of the segments in seconds.
        sampling_rate (int): sampling rate of the data.

    Returns:
        np.array: segmented data.
    """
    n_samples = sampling_rate * segment_size
    n_segments = len(data) // n_samples

    return np.array([data[i * n_samples:(i + 1) * n_samples] for i in range(n_segments)])


def retrive_data(recordings, in_path, start_datetime, end_datetime, n_channels = 23):
    """
    Retrieve the data of the patient between the specified datetimes.

    Args:
        recordings (list): list of EEGrec objects.
        in_path (str): path where the recordings are stored.
        start_datetime (datetime): start datetime.
        end_datetime (datetime): end datetime.
        n_channels (int, optional): number of channels. Defaults to 23.

    Returns:
        np.array: data of the patient between the specified datetimes.
    """

    for i, rec in enumerate(recordings):
        if rec.start <= start_datetime <= rec.end:
            if rec.start <= end_datetime <= rec.end:
                return rec.retrive_data(f'{in_path}/{rec.id}.edf', 
                                        start_seconds = (start_datetime - rec.start).total_seconds(), 
                                        end_seconds = (end_datetime - rec.start).total_seconds(),
                                        n_channels=n_channels)
            else:             
                data = rec.retrive_data(f'{in_path}/{rec.id}.edf', start_seconds = (start_datetime - rec.start).total_seconds(), n_channels=n_channels)
                break
        
    for rec in recordings[i+1:]:
        if rec.start <= end_datetime <= rec.end:
            rec_data = rec.retrive_data(f'{in_path}/{rec.id}.edf', end_seconds = (end_datetime - rec.start).total_seconds(), n_channels=n_channels)
            data = np.vstack((data, rec_data))
            return data
        else:
            rec_data = rec.retrive_data(f'{in_path}/{rec.id}.edf', n_channels=n_channels)
            data = np.vstack((data, rec_data))


def check_datetime(recordings, phase_datetime, control = 'start'):
    """
    Check if the phase datetime is within the recordings. If not, return the closest datetime.

    Args:
        recordings (list): list of EEGrec objects.
        phase_datetime (datetime): datetime to check.
        control (str, optional): control to return the closest datetime. Defaults to 'start'.

    Returns:
        datetime: closest datetime to the phase datetime.
    """
    min_diff = float('inf')
    min_index = None

    for i, rec in enumerate(recordings):
        if rec.start <= phase_datetime <= rec.end:
            return phase_datetime

        diff = abs(rec.start - phase_datetime).total_seconds()
        if diff < min_diff:
            min_diff = diff
            min_index = i

    if control.lower() == 'end' and min_index > 0:
        return recordings[min_index - 1].end

    return recordings[min_index].start


def get_phase_datetimes(recordings, reference_datetime, duration, mod, gap = 0):
    """
    Get the start and end datetimes of the phase.

    Args:
        recordings (list): list of EEGrec objects.
        reference_datetime (datetime): datetime to reference.
        duration (int): duration of the phase in hours.
        mod (str): mode to get the datetimes.
        gap (int, optional): gap in seconds. Defaults to 0.

    Returns:
        _type_: _description_
    """
    end = check_datetime(recordings, reference_datetime - timedelta(seconds = gap), control = 'end')
    
    if mod.lower() == 'static':
        start = check_datetime(recordings, end - timedelta(hours = duration), control = 'start')
    elif mod.lower() == 'dynamic':
        reversed_recs = recordings[::-1]
        for i, rec in enumerate(reversed_recs):
            if rec.start <= end <= rec.end:
                duration -= (end - rec.start).total_seconds() / 3600
                break

        if duration > 0:
            for rec in reversed_recs[i+1:]:
                duration -= (rec.end - rec.start).total_seconds() / 3600
                if duration <= 0:
                    break
        
        start = rec.start + timedelta(hours = abs(duration)) if duration < 0 else rec.start
    
    return start, end