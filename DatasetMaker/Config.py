"""Fundamental Configuration for Data Preprocessing"""
"""
Workspace
"""
WORKSPACE_PATH = '/root/autodl-pub/YYF/Large_EEG_model/EEGdata'

"""
Root path of raw data of Temple University Hospital EEG Corpus.
You can change the root path after you download the data from:
https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""
RAW_DATA_ROOT_PATH = '/root/autodl-pub/YYF/EEGdata'

"""
Montage path for eeg data loading from EDF files.
"""
MONTAGE_PATH = '/root/autodl-pub/YYF/Large_EEG_model/EEGdata/DOCS'

"""
Six types of datasets can be chosen for different tasks. Their information, 
provided by [https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml]
is as following:

1. The TUH Abnormal EEG Corpus (TUAB) [59GB]: A corpus of EEGs that have 
been annotated as normal or abnormal. Read Silvia Lopez's MS thesis 
(https://isip.piconepress.com/publications/ms_theses/2017/abnormal/thesis/) 
for a description of the corpus.

2. The TUH EEG Artifact Corpus (TUAR) [5.4GB]: This subset of TUEG that 
contains annotations of 5 different artifacts: (1) eye movement (EYEM), 
(2) chewing (CHEW), (3) shivering (SHIV), (4) electrode pop, electrode 
static, and lead artifacts (ELPP), and (5) muscle artifacts (MUSC).

3. The TUH EEG Epilepsy Corpus (TUEP) [36GB]: This is a subset of TUEG 
that contains 100 subjects epilepsy and 100 subjects without epilepsy, 
as determined by a certified neurologist. The data was developed in 
collaboration with a number of partners including NIH.

4. The TUH EEG Events Corpus (TUEV) [19GB]: This corpus is a subset of 
TUEG that contains annotations of EEG segments as one of six classes: 
(1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform 
discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), 
(4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).

5. The TUH EEG Seizure Corpus (TUSZ) [79GB]: This corpus has EEG signals 
that have been manually annotated data for seizure events (start time, stop, 
channel and seizure type). For more information about this corpus, please refer to 
the book section (https://isip.piconepress.com/publications/book_sections/2018/frontiers_neuroscience/tuh_eeg/). 
Our annotation guidelines are described here (https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/_DOCS/annotation).

6. The TUH EEG Slowing Corpus (TUSL) [1.6GB]: This is another subset of TUEG 
that contains annotations of slowing events. This corpus has been used to study 
common error modalities in automated seizure detection.
"""
DATA_SET_NAME = 'TUAB'

"""
File for saving the detail information of each dataset. 
"""
DATA_INFO_SAVER = 'Info_{}.csv'.format(DATA_SET_NAME)

"""
Random seed for dataset split
"""
DATASET_RANDOM_SEED = 1024


"""
A map from channel name to channel id 
"""
CHANNEL_TO_ID = {
    'FP1-F7': 0,
    'F7-T3': 1,
    'T3-T5': 2,
    'T5-O1': 3,
    'FP2-F8': 4,
    'F8-T4': 5,
    'T4-T6': 6,
    'T6-O2': 7,
    'A1-T3': 8,
    'T3-C3': 9,
    'C3-CZ': 10,
    'CZ-C4': 11,
    'C4-T4': 12,
    'T4-A2': 13,
    'FP1-F3': 14,
    'F3-C3': 15,
    'C3-P3': 16,
    'P3-O1': 17,
    'FP2-F4': 18,
    'F4-C4': 19,
    'C4-P4': 20,
    'P4-O2': 21,
    'EKG': 22
}

"""
TUSZ: A map from seizure class to class id
"""
SEIZURE_TO_ID = {
    'bckg': 0,
    'seiz': 1,
    'fnsz': 2,
    'gnsz': 3,
    'spsz': 4,
    'cpsz': 5,
    'absz': 6,
    'tnsz': 7,
    'cnsz': 8,
    'tcsz': 9,
    'atsz': 10,
    'mysz': 11,
    'nesz': 12
}
'''
"TUEV": A map from label name to label id
'''
EVENT_TO_ID = {
    'spsw': 1,
    'gped': 2,
    'pled': 3,
    'eyem': 4,
    'artf': 5,
    'bckg': 0,
}


'''
"TUSL": A map from label name to label id
'''
SLOW_TO_ID = {
    'bckg': 0,
    'seiz': 1,
    'slow': 2,
}


'''
"TUAR": A map from label name to label id
'''
ARTIFACT_TO_ID = {
    'null': 0,
    'eyem': 1,
    'chew': 2,
    'shiv': 3,
    'musc': 4,
    'elpp': 5,
    'elec': 6,
    'eyem_chew': 7,
    'eyem_shiv': 8,
    'eyem_musc': 9,
    'eyem_elec': 10,
    'chew_musc': 11,
    'chew_elec': 12,
    'shiv_elec': 13,
    'musc_elec': 14
}