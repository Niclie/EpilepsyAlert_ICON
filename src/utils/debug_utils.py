import os
from src.data_preprocessing import preprocess


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_duration(recordings, start_datetime, end_datetime):
    """
    Calculate the duration between two datetimes in hours based on the data in the recordings.

    Args:
        recordings (list): list of EEGRec.
        start_datetime (datetime): start datetime.
        end_datetime (datetime): end datetime.

    Returns:
        float: duration in hours.
    """
    seconds = 0
    for i, rec in enumerate(recordings):
        if rec.start <= start_datetime <= rec.end:
            if rec.start <= end_datetime <= rec.end:
                return (end_datetime - start_datetime).total_seconds()
            
            seconds += (rec.end - start_datetime).total_seconds()
            break

    for rec in recordings[i+1:]:
        if rec.start <= end_datetime <= rec.end:
            seconds += (end_datetime - rec.start).total_seconds()
            break
        seconds += (rec.end - rec.start).total_seconds()
        
    hours, remainder = divmod(seconds, 3600)  # Ottiene le ore e i secondi rimanenti
    minutes, seconds = divmod(remainder, 60)  # Ottiene i minuti e i secondi rimanenti
    
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def get_recordings_gap(patient):
    """
    Get the gap between recordings in seconds.

    Args:
        patient (Patient): patient object.

    Returns:
        list: list of gaps between recordings in seconds.
    """
    recordings = preprocess.min_channel_recordings(patient, 23)

    gap = []
    for recs in recordings:
        for i, r in enumerate(recs[1:], 1):
            gap.append((r.start - recs[i-1].end).total_seconds())
            
    return gap


def get_interictal_preictal_duration(patient):
    """
    Get the duration of interictal + preictal phases in hours.

    Args:
        patient (Patient): patient object.

    Returns:
        list: list of durations in hours
    """
    recordings = preprocess.min_channel_recordings(patient, 23)

    d = []
    for recs in recordings:
        last_seizure_end = recs[0].start
        for r in recs:
            for s in r.get_seizures_datetimes():
                d.append(round((s[0] - last_seizure_end).total_seconds() / 3600, 3))
                last_seizure_end = s[1]

    return d


def get_seizures_datetimes(patient):
    """
    Get the datetimes of seizures.

    Args:
        patient (Patient): patient object.

    Returns:
        list: list of seizures datetimes.
    """
    sd = []
    for r in patient.recordings:
        for s in r.get_seizures_datetimes():
            sd.append(s)
            
    return sd


def print_recordings_dataset(patient):
    """
    Print a rappresentation of the dataset of the specified patient.

    Args:
        patient (Patient): patient object.
    """
    grouped_recordings = preprocess.min_channel_recordings(patient, 23)
    for recordings in grouped_recordings:
        prev_seizure_end = recordings[0].start
        for rec in recordings:
            for s in rec.get_seizures_datetimes():
                start_preictal, end_preictal = preprocess.get_phase_datetimes(recordings, s[0], 1, mod = 'static', gap = 1)
                start_interictal, end_interictal = preprocess.get_phase_datetimes(recordings, start_preictal, 4, mod = 'dynamic', gap = 0)
                
                if not(start_interictal <= prev_seizure_end <= end_preictal):
                    print(f'Seizure at {rec.id}:\nstart: {s[0]} -> end: {s[1]}\n')
                    print(f'Preictal: {start_preictal} -> {end_preictal}\n')
                    print(f'Interictal: {start_interictal} -> {end_interictal}\n\n')
                prev_seizure_end = s[1]