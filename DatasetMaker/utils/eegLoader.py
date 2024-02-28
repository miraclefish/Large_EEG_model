from pathlib import Path
import pandas as pd
import mne
import os


drop_channels = ['PHOTIC-REF', 'PHOTIC PH', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF',
                 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF',
                 'EEG SP2-REF', 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF',
                 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF',
                 'EEG PG1-REF', 'EEG OZ-REF']
drop_channels.extend([channel.split('-')[0] + '-LE' for channel in drop_channels if channel[-3:] == 'REF'])
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
drop_channels.extend([f'EEG {i}-LE' for i in range(20, 129)])
drop_channels.extend([f'DC{i}-DC' for i in range(0, 10)])

# unified_montage : 23 channels
chOrder_unified = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                   'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']

# ar_montage : 23 channels
chOrder_standard_ar = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
                       'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
                       'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
                       'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
                       'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

# le_montage : 23 channels
chOrder_standard_le = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
                       'EEG C4-LE', 'EEG A1-LE', 'EEG A2-LE', 'EEG P3-LE', 'EEG P4-LE',
                       'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE',
                       'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE',
                       'EEG PZ-LE', 'EEG T1-LE', 'EEG T2-LE']

# ar_a_montage : 19 channels
chOrder_standard_ar_a = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
                         'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
                         'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
                         'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

chOrder_standard = {
    'ar': chOrder_standard_ar,
    'ar_a': chOrder_standard_ar_a,
    'le': chOrder_standard_le
}

class MontageBuilder(object):

    def __init__(self, path):

        self.path = path
        self.montage_types = ['ar', 'le', 'ar_a', 'le_a']
        self.montage_files = self.get_montage_files()

    def initial_montage(self):
        montage_dict = {}
        for m, f in zip(self.montage_types, self.montage_files):
            montage = self.load_montage(f)
            montage_dict[m] = montage
        return montage_dict

    def get_montage_files(self):

        files = []
        file_list = os.listdir(self.path)
        for montage in self.montage_types:
            for file in file_list:
                if montage in file:
                    files.append(os.path.join(self.path, file))
                    break
        return files

    def load_montage(self, path):
        montage = {}
        f = open(path)

        for line in f.readlines():
            if len(line) < 20 or line[0] == '#':
                continue
            colon_split = line.split(':')
            channel_name = colon_split[0].split(' ')[-1]
            channel_split = colon_split[1].split('--')
            if len(channel_split) == 1:
                montage[channel_name] = channel_split[0].strip()
            else:
                montage[channel_name] = (channel_split[0].strip(), channel_split[1].strip())

        return montage

class EDFReader(object):

    def __init__(self, file_path, montage_type, l_freq, h_freq, rsfreq):
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.montage_type = montage_type
        self.rsfreq = rsfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.raw_data, self.L, self.freq, self.duration = self.load_raw_eeg()
        self.processed_data = self.preprocess_eeg(self.raw_data)


    def preprocess_eeg(self, raw):

        # processor = Preprocessor(self.raw_eeg, self.freq, self.target_freq)
        # eeg = processor.implement()
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
        raw.notch_filter(50.0)
        raw.resample(self.rsfreq, n_jobs=5)
        ch_name = [name.split('-')[0][4:] for name in raw.ch_names]
        eeg_data = raw.get_data(units='uV')
        processed_data = pd.DataFrame(eeg_data.T, columns=ch_name)

        if ch_name != chOrder_unified:
            for name in chOrder_unified:
                if name not in ch_name:
                    processed_data[name] = 0
            processed_data = processed_data[chOrder_unified]

        return processed_data

    def load_raw_eeg(self):
        try:
            raw = self.load_by_mne(self.file_path)
            freq = int(raw.info['sfreq'])
            L = raw.n_times
            duration = int(L / freq)
            return raw, L, freq, duration
        except:
            with open(f"process-error-files.txt", "a") as f:
                f.write(self.file_path.name + ": " + str(self.file_path) + "\n")
            return None
        # if L // freq != duration:
        #     print("Error: {} in [{}] loading error !".format(self.file_name, self.file_path))
        #     print("The duration [{}s] is not corresponding to the sampling length [{}] with {}Hz".format(duration, L,
        #                                                                                                  freq))
        #     sys.exit(1)
        #
        # montage_data = {}
        #
        # for montage_channel, channel_pair in self.montage.items():
        #     if len(channel_pair) > 2:
        #         montage_data[montage_channel] = raw_data_dict[channel_pair]
        #     else:
        #         montage_data[montage_channel] = raw_data_dict[channel_pair[0]] - raw_data_dict[channel_pair[1]]
        #
        # montage_signal = pd.DataFrame(montage_data)
        #
        # for channel in ['A1-T3', 'T4-A2', 'EKG']:
        #     if channel not in montage_signal.columns:
        #         montage_signal[channel] = np.zeros(montage_signal.shape[0])
        #
        # columns = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
        #            'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
        #            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'EKG']
        #
        # montage_signal = montage_signal[columns]
        # montage_signal = montage_signal.round(2)


    def load_by_mne(self, path):
        raw = mne.io.read_raw_edf(path, preload=True)
        if drop_channels is not None:
            useless_chs = []
            for ch in drop_channels:
                if ch in raw.ch_names:
                    useless_chs.append(ch)
            raw.drop_channels(useless_chs)
        if chOrder_standard.get(self.montage_type) and len(chOrder_standard[self.montage_type]) == len(raw.ch_names):
            raw.reorder_channels(chOrder_standard[self.montage_type])
        if raw.ch_names != chOrder_standard[self.montage_type]:
            for name in raw.ch_names:
                if name not in chOrder_standard[self.montage_type]:
                    if name not in chOrder_standard_ar:
                        raise Exception("channel order is wrong!")
        return raw

    # def load_by_pyedflib(self, path):
    #     raw_edf = pyedf.EdfReader(path)
    #     freq = raw_edf.getSampleFrequencies()
    #     duration = raw_edf.file_duration
    #     L = raw_edf.getNSamples()
    #     channel_name = raw_edf.getSignalLabels()
    #     raw_data_dict = {}
    #     for i, channel in enumerate(channel_name):
    #         if channel in ['EEG T1-REF', 'EEG T2-REF']:
    #             channel = f'{channel[:4]}A{channel[5:]}'
    #         raw_data_dict[channel] = raw_edf.readSignal(i)
    #
    #     return raw_data_dict, int(L[0]), int(freq[0]), duration