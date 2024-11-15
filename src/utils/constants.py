#This file contains all the constants used in the project.
PATIENTS = ('chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19','chb20', 'chb21', 'chb22', 'chb23')

DATA_FOLDER                 = 'data/raw'
DATASETS_FOLDER             = 'data/converted'
PREPROCESSED_FOLDER         = 'data/preprocessed'

RESULTS_FOLDER              = 'results'
MODELS_FOLDER               = 'results/models'
PLOTS_FOLDER                = 'results/plots'

TIME_FORMAT                 = '%H:%M:%S'

REGEX_FILE_INFO_PATTERN     = r'File Name.*\nFile Start.*\nFile End.*\nNumber of Seizures.*\n(?:Seizure (?:\d+ )?Start.*\nSeizure (?:\d+ )?End.*\n?)*'
REGEX_CHANNEL_SELECTOR      = r'Channel \d*:\s*(.*)'
REGEX_BASE_INFO_SELECTOR    = r'File Name:\s*(.*).edf\nFile Start Time:\s*(.*)\nFile End Time:\s*(.*)\nNumber of Seizures in File:\s*(\d+)'
REGEX_SEIZURE_INFO_SELECTOR = r'Seizure (?:\d+ )?Start Time:\s*(\d+) seconds\nSeizure (?:\d+ )?End Time:\s*(\d+) seconds'