import os
import re
from datetime import datetime, timedelta
from src.utils import constants
from src.data_preprocessing.eeg_recording import EEGRec
from src.data_preprocessing.patient import Patient
from src.data_preprocessing.seizure import Seizure


def convert_time(time_str, last_date, time_format = constants.TIME_FORMAT):
    """
    Convert the time string to a datetime object considering the last date for determining if the time is from the next day.

    Args:
        time_str (str): time string to convert.
        last_date (datetime): last known date for determining if the time is from the next day.
        time_format (str, optional): format of the time string. Defaults to constants.TIME_FORMAT.

    Returns:
        datetime: datetime object with the converted time.
    """

    hour = int(time_str.split(':')[0])
    if hour >= 24:
        hour = f'{(hour - 24):02d}'
        converted_date = datetime.strptime(hour + ':' + time_str[3:], time_format)
    else:
        converted_date = datetime.strptime(time_str, time_format)
    
    converted_date = converted_date.replace(year=last_date.year, month=last_date.month, day=last_date.day)
    if converted_date.time() < last_date.time():
        converted_date += timedelta(days=1)
    
    return converted_date


def load_summary_from_file(p_id,
                           data_folder        = constants.DATA_FOLDER,
                           time_format        = constants.TIME_FORMAT,
                           channels_selector  = re.compile(constants.REGEX_CHANNEL_SELECTOR),
                           file_info_pattern  = re.compile(constants.REGEX_FILE_INFO_PATTERN),
                           base_info_selector = re.compile(constants.REGEX_BASE_INFO_SELECTOR),
                           seizure_selector   = re.compile(constants.REGEX_SEIZURE_INFO_SELECTOR)):
    """
    Load the summary of the EEG recordings from the specified file. The file should be named as {p_id}-summary.txt.

    Args:
        p_id (str): patient ID.
        data_folder (str, optional): folder where the data is stored. Defaults to constants.DATA_FOLDER.
        time_format (str, optional): format of datetime. Defaults to constants.TIME_FORMAT.
        channels_selector (str, optional): regex to select the channels. Defaults to re.compile(constants.REGEX_CHANNEL_SELECTOR).
        file_info_pattern (str, optional): regex to select the sections that contain the information of each recording. Defaults to re.compile(constants.REGEX_FILE_INFO_PATTERN).
        base_info_selector (str, optional): regex to select the base information of each recording. Defaults to re.compile(constants.REGEX_BASE_INFO_SELECTOR).
        seizure_selector (str, optional): regex to select the seizure information. Defaults to re.compile(constants.REGEX_SEIZURE_INFO_SELECTOR).

    Returns:
        Patient: patient with the EEG recordings.
    """
    
    with open(os.path.join(data_folder, p_id, f'{p_id}-summary.txt'), 'r') as file:
        content = re.split(r'\*+', file.read())

    sampling_rate = int(content[0].split()[3])

    last_date = datetime.strptime('00:00:00', time_format)
    rec_info = []
    for split in content[2:]:
        for file_info in file_info_pattern.findall(split):
            id, rec_start, rec_end, n_seizures = base_info_selector.search(file_info).group(1, 2, 3, 4)
            rec_start, rec_end = map(lambda t: convert_time(t, last_date), (rec_start, rec_end))
            seizures = [Seizure(f'{id}_{i}', rec_start + timedelta(seconds=int(s_start)), rec_start + timedelta(seconds=int(s_end))) for i, (s_start, s_end) in enumerate(seizure_selector.findall(file_info), start=1)]
            channels = tuple(filter(lambda x: x != '-' , channels_selector.findall(split)))

            rec_info.append(EEGRec(id, rec_start, rec_end, seizures, channels, sampling_rate))
            
            last_date = rec_end
            
    return Patient(p_id, rec_info)


def load_summaries_from_folder(data_folder = constants.DATA_FOLDER, time_format = constants.TIME_FORMAT, exclude = None):
    """
    Load the summaries of the EEG recordings from the specified folder. The files should be named as {p_id}-summary.txt.

    Args:
        data_folder (str, optional): folder where the data is stored. Defaults to constants.DATA_FOLDER.
        time_format (str, optional): format of datetime. Defaults to constants.TIME_FORMAT.
        exclude (list, optional): list of patient IDs to exclude. Defaults to None.

    Returns:
        list: list of patients with their EEG recordings.
    """

    p_id_list = next(os.walk(data_folder))[1]

    if exclude is not None:
        p_id_list = [p_id for p_id in p_id_list if p_id not in exclude]

    channels_selector  = re.compile(constants.REGEX_CHANNEL_SELECTOR)
    file_info_pattern  = re.compile(constants.REGEX_FILE_INFO_PATTERN)
    base_info_selector = re.compile(constants.REGEX_BASE_INFO_SELECTOR)
    seizure_selector   = re.compile(constants.REGEX_SEIZURE_INFO_SELECTOR)

    return [load_summary_from_file(id, data_folder, time_format, channels_selector, file_info_pattern, base_info_selector, seizure_selector) for id in p_id_list]