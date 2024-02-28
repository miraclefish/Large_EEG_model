import os
import sys
from scipy.signal import butter, filtfilt
from scipy import interpolate
from pathlib import Path
import pyedflib as pyedf
import pandas as pd
import numpy as np
import random

from tqdm import tqdm
from Config import DATASET_RANDOM_SEED


def find_files_with_extension(folder_path, extension):
    file_list = []
    file_path_list = []
    extension_length = len(extension)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_path_list.append(file_path)
                file_list.append(file[:-extension_length])
    file_list.sort()
    file_path_list.sort()
    return file_list, file_path_list


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


class Preprocessor(object):

    def __init__(self, raw_eeg, source_freq, target_freq, label=None):

        self.raw_eeg = raw_eeg
        self.source_freq = source_freq
        self.target_freq = target_freq
        self.label = label

    def implement(self):

        eeg, label = self.remove_zero_segments(self.raw_eeg, self.source_freq, self.label)

        # band filter 0.5 - 80 Hz
        data = self.band_filter(eeg.values, 0.3, 40, self.source_freq)
        # north filter 60 Hz
        data = self.north_filter(data, 59, 61, self.source_freq)
        # 裁切数据在 -500 ~ 500 uv 之间
        data = np.clip(data, -500, 500)

        data = self.downsample(data, self.source_freq, self.target_freq)

        processed_eeg = pd.DataFrame(data, columns=eeg.columns)

        return processed_eeg

    def downsample(self, data, freq, target_freq):
        # 确定降采样比例
        original_length = data.shape[0]
        time_original = np.arange(0, original_length) / freq

        target_length = int(original_length / freq * target_freq)
        time_target = np.arange(0, target_length) / target_freq

        downsampled_data = np.empty((len(time_target), data.shape[1]))

        # 使用线性插值进行降采样
        for i in range(data.shape[1]):
            f = interpolate.interp1d(time_original, data[:, i], kind='linear')
            downsampled_data[:, i] = f(time_target)

        return downsampled_data

    def remove_zero_segments(self, df, l_th, label=None):
        # df is the raw eeg data, and the index is its time index, the column is its channel dimension.
        # df = pd.DataFrame(...)

        # find the start and end index of zeros segments in the raw eeg signals.

        zero_mask = df.eq(0).astype(int)  # build a bool mask the same shape with df, indicating which element is zero.
        changes = zero_mask.diff()  # find the change points in the mask.

        start_mask = changes.eq(1)  # the mask of start indices of zero segments
        end_mask = changes.eq(-1)  # teh mask of end indices of zero segments

        start_indices = start_mask.index[start_mask.all(axis=1)].tolist()  # the start indices of zero segments
        end_indices = end_mask.index[end_mask.all(axis=1)].tolist()  # the end indices of zero segments

        start_end_flags = np.concatenate([['s'] * len(start_indices), ['e'] * len(end_indices)])
        start_end_indices = np.concatenate([start_indices, end_indices])

        sort_index = np.argsort(start_end_indices)
        start_end_indices = start_end_indices[sort_index]
        start_end_flags = start_end_flags[sort_index]

        start_end_indices = np.concatenate([[0, ], start_end_indices, [zero_mask.shape[0], ]])
        start_end_flags = np.concatenate([['s', ], start_end_flags, ['e', ]])

        start = []
        end = []

        for i in range(len(start_end_indices) - 1):
            if start_end_flags[i] == 's' and start_end_flags[i + 1] == 'e':
                if zero_mask[int(start_end_indices[i]):int(start_end_indices[i + 1])].all().all():
                    start.append(int(start_end_indices[i]))
                    end.append(int(start_end_indices[i + 1]))

        zero_segments = list(zip(start, end))  # Pairs the start and end indices into tuples

        # Print the start and end indices of the all zeros signal segments
        attribute = {0: 'Head', 1: 'Tail', 2: 'Inter'}
        count = 0
        final_show = False
        for segment in zero_segments:
            flag = 2
            show = False
            if segment[0] == 0:
                flag = 0
            if segment[1] == zero_mask.shape[0]:
                flag = 1
            if (flag == 0 or flag == 1) and segment[1] - segment[0] >= l_th:
                show = True
                count += 1
                final_show = True
                df.drop(list(np.arange(segment[0], segment[1])), inplace=True)
                if flag == 0:
                    if label is not None:
                        print('Resume label index...')
                        label.loc[:, ['start', 'end']] = label.loc[:, ['start', 'end']] - (segment[1] - segment[0])
            if show:
                print("Start: {}, End: {}, L: {}, Attribute: {}".format(segment[0], segment[1],
                                                                        segment[1] - segment[0],
                                                                        attribute[flag]))

        df = df.reset_index(drop=True)

        if final_show:
            print(
                'Original Length [{}] - Now Length [{}] = Drop Length [{}]'.format(zero_mask.shape[0], df.shape[0],
                                                                                   zero_mask.shape[0] - df.shape[
                                                                                       0]))
        return df, label

    def band_filter(self, signal, lowcut, highcut, fs):
        '''
        fs: Sampling Frequency
        lowcut: low cut frequency
        highcut: high cut frequency
        '''
        # fs = 250.0  # Hz

        # Butterworth filter
        nyquist = 0.5 * fs  # Nyquist frequency（half of sampling frequency）
        low = lowcut / nyquist
        high = highcut / nyquist
        order = 1  # order of the filter
        [b, a] = butter(order, [low, high], btype='band')

        # implement the filter
        filtered_signal = filtfilt(b, a, signal, axis=0)
        return filtered_signal

    def north_filter(self, signal, lowcut, highcut, fs):
        '''
        fs: Sampling Frequency
        lowcut: low cut frequency
        highcut: high cut frequency
        '''
        # fs = 250.0  # Hz

        # Butterworth filter
        nyquist = 0.5 * fs  # Nyquist frequency（half of sampling frequency）
        low = lowcut / nyquist
        high = highcut / nyquist
        order = 4  # order of the filter
        [b, a] = butter(order, [low, high], btype='bandstop')

        # implement the filter
        filtered_signal = filtfilt(b, a, signal, axis=0)
        return filtered_signal


class EDFReader(object):

    def __init__(self, file_path, montage_type, target_freq):
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.montage_type = montage_type
        self.target_freq = target_freq
        self.raw_eeg, self.L, self.freq, self.duration = self.load_raw_eeg()
        self.processed_data = self.preprocess_eeg()


    def preprocess_eeg(self):

        processor = Preprocessor(self.raw_eeg, self.freq, self.target_freq)
        eeg = processor.implement()

        return eeg

    def load_raw_eeg(self):

        raw_data_dict, L, freq, duration = self.load_by_pyedflib(self, self.file_path)

        if L // freq != duration:
            print("Error: {} in [{}] loading error !".format(self.file_name, self.file_path))
            print("The duration [{}s] is not corresponding to the sampling length [{}] with {}Hz".format(duration, L,
                                                                                                         freq))
            sys.exit(1)

        montage_data = {}

        for montage_channel, channel_pair in self.montage.items():
            if len(channel_pair) > 2:
                montage_data[montage_channel] = raw_data_dict[channel_pair]
            else:
                montage_data[montage_channel] = raw_data_dict[channel_pair[0]] - raw_data_dict[channel_pair[1]]

        montage_signal = pd.DataFrame(montage_data)

        for channel in ['A1-T3', 'T4-A2', 'EKG']:
            if channel not in montage_signal.columns:
                montage_signal[channel] = np.zeros(montage_signal.shape[0])

        columns = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
                   'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
                   'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'EKG']

        montage_signal = montage_signal[columns]
        montage_signal = montage_signal.round(2)

        return montage_signal

    def load_by_pyedflib(self, path):
        raw_edf = pyedf.EdfReader(path)
        freq = raw_edf.getSampleFrequencies()
        duration = raw_edf.file_duration
        L = raw_edf.getNSamples()
        channel_name = raw_edf.getSignalLabels()
        raw_data_dict = {}
        for i, channel in enumerate(channel_name):
            if channel in ['EEG T1-REF', 'EEG T2-REF']:
                channel = f'{channel[:4]}A{channel[5:]}'
            raw_data_dict[channel] = raw_edf.readSignal(i)

        return raw_data_dict, int(L[0]), int(freq[0]), duration


class Reader(object):

    def __init__(self, path, info_saver, montage, target_freq):
        self.root_path = path
        self.montage = montage
        self.info_saver = info_saver
        self.target_freq = target_freq
        self.file_list = []

    def __len__(self):
        return len(self.file_list)

    def load_info_from_saver(self):
        INFO = pd.read_csv(self.info_saver, index_col=0)
        return list(INFO['name'].values), INFO

    def restore_file_info(self, INFO):
        INFO.to_csv(self.info_saver)
        return None


class ReaderTUAB(Reader):

    def __init__(self, path, info_saver, montage, target_freq):

        super(ReaderTUAB, self).__init__(path, info_saver, montage, target_freq)

        if os.path.exists(self.info_saver):
            self.file_list, self.file_info = self.load_info_from_saver()
        else:
            self.file_list, self.file_info = self.get_all_file_info()

    def __getitem__(self, id):
        eeg_item = self.load_data(id)
        label = self.file_info.loc[id, 'type']
        folder = self.file_info.loc[id, 'sub_folder']
        return eeg_item, label, folder

    def load_data(self, id):
        edf_reader = EDFReader(file_path=self.file_info.loc[id, 'path'], montage=self.montage['ar'],
                               target_freq=self.target_freq)
        return edf_reader

    def get_all_file_info(self):

        all_file_list = []
        all_file_path = []
        all_file_info = []
        count = np.zeros((3, 3))
        for i, sub_folder in enumerate(['eval', 'train']):
            for j, file_type in enumerate(['normal', 'abnormal']):
                file_list, file_path, file_info = self._get_file_list(sub_folder, file_type)
                count[i, j] = len(file_list)
                all_file_list += file_list
                all_file_path += file_path
                all_file_info += file_info

        for i, path in enumerate(all_file_path):
            raw_edf = pyedf.EdfReader(path)
            all_file_info[i].append(int(raw_edf.getSampleFrequencies()[0]))
            all_file_info[i].append(int(raw_edf.getNSamples()[0]))
            all_file_info[i].append(int(raw_edf.file_duration))

        count[:, -1] = np.sum(count[:, :2], axis=1)
        count[-1, :] = np.sum(count[:2, :], axis=0)
        count = pd.DataFrame(count, columns=['Normal', 'Abnormal', 'Total'],
                             index=['Evaluation', 'Train', 'Total'])

        # Format the output
        # pd.options.display.float_format = '{:,.0f}'.format
        INFO = pd.DataFrame(all_file_info,
                            columns=['name', 'sub_folder', 'type', 'subject',
                                     'session', 'file', 'path', 'freq', 'L', 'duration'])

        duration = np.zeros((3, 3))
        for i, sub_folder in enumerate(['eval', 'train']):
            for j, file_type in enumerate(['normal', 'abnormal']):
                duration[i, j] = INFO[(INFO['sub_folder'] == sub_folder) & (INFO['type'] == file_type)].loc[:,
                                 'duration'].sum() / 3600
        duration[:, -1] = np.sum(duration[:, :2], axis=1)
        duration[-1, :] = np.sum(duration[:2, :], axis=0)
        duration = pd.DataFrame(duration, columns=['Normal', 'Abnormal', 'Total'],
                                index=['Evaluation', 'Train', 'Total'])

        # Print the No. of Files
        print("| Size (No. of files)")
        print("|----------------------------------------------------------------------|")
        print("| Description |      Normal      |     Abnormal     |      Total       |")
        print("|-------------+------------------+------------------+------------------|")
        for index, row in count.iterrows():
            print(f"| {index:<11} | {int(row['Normal']):^16} | {int(row['Abnormal']):^16} | {int(row['Total']):^16} |")
            print("|-------------+------------------+------------------+------------------|")

        # Print Hours of data
        print("| Size (Hours of data)")
        print("|----------------------------------------------------------------------|")
        print("| Description |      Normal      |     Abnormal     |      Total       |")
        print("|-------------+------------------+------------------+------------------|")
        for index, row in duration.iterrows():
            print(
                f"| {index:<11} | {row['Normal']:^16.2f} | {row['Abnormal']:^16.2f} | {row['Total']:^16.2f} |")
            print("|-------------+------------------+------------------+------------------|")

        """ Reference:
        Size (No. of Files / Hours of Data):
             |----------------------------------------------------------------------|
             | Description |      Normal      |     Abnormal     |      Total       |
             |-------------+------------------+------------------+------------------|
             | Evaluation  |   150 (   55.46) |   126 (   47.48) |   276 (  102.94) |
             |-------------+------------------+------------------+------------------|
             | Train       | 1,371 (  512.01) | 1,346 (  526.05) | 2,717 (1,038.06) |
             |-------------+------------------+------------------+------------------|
             | Total       | 1,521 (  567.47) | 1,472 (  573.53) | 2,993 (1,142.00) |
             |----------------------------------------------------------------------|
        """

        self.restore_file_info(INFO)

        return all_file_list, INFO

    def _get_file_list(self, sub_folder, file_type):
        folder = '{}/edf/{}/{}/01_tcp_ar'.format(self.root_path, sub_folder, file_type)
        file_list = [file[:-4] for file in os.listdir(folder)]
        file_path = [os.path.join(folder, file + '.edf') for file in file_list]
        file_info = [[file,
                      sub_folder,
                      file_type,
                      file.split('_')[0],
                      file.split('_')[1],
                      file.split('_')[2],
                      path] for file, path in zip(file_list, file_path)]

        return file_list, file_path, file_info


class ReaderTUEP(Reader):

    def __init__(self, path, info_saver, montage, target_freq, eval_ratio=0.1):

        super(ReaderTUEP, self).__init__(path, info_saver, montage, target_freq)
        self.eval_ratio = eval_ratio
        if os.path.exists(self.info_saver):
            self.file_list, self.file_info = self.load_info_from_saver()
        else:
            self.file_list, self.file_info = self.get_all_file_info()
        if 'sub_folder' not in self.file_info.columns:
            self.prepare_train_and_eval()

    def __getitem__(self, id):
        eeg_item = self.load_data(id)
        label = self.file_info.loc[id, 'type']
        folder = self.file_info.loc[id, 'sub_folder']
        return eeg_item, label, folder

    def load_data(self, id):
        edf_reader = EDFReader(file_path=self.file_info.loc[id, 'path'],
                               montage=self.montage[self.file_info.loc[id, 'montage']],
                               target_freq=self.target_freq)
        return edf_reader

    def prepare_train_and_eval(self):
        subject_list = list(self.file_info['subject'].value_counts().index)

        random.seed(DATASET_RANDOM_SEED)
        random.shuffle(subject_list)

        split_index = int(len(subject_list) * (1 - self.eval_ratio))
        train_list = subject_list[:split_index]
        eval_list = subject_list[split_index:]

        def assign_group(row):
            if row['subject'] in train_list:
                return 'train'
            elif row['subject'] in eval_list:
                return 'eval'
            else:
                return 'Unknown'

        self.file_info['sub_folder'] = self.file_info.apply(assign_group, axis=1)
        new_index = 1
        label_column = self.file_info.pop('sub_folder')
        self.file_info.insert(new_index, 'sub_folder', label_column)

        self.restore_file_info(self.file_info)

        return None

    def get_all_file_info(self):

        all_file_list = []
        all_file_path = []
        all_file_info = []
        for i, type in enumerate(['epilepsy', 'no_epilepsy']):
            file_list, file_path, file_info = self._get_file_list(type)
            all_file_list += file_list
            all_file_path += file_path
            all_file_info += file_info

        for i, path in enumerate(all_file_path):
            raw_edf = pyedf.EdfReader(path)
            all_file_info[i].append(int(raw_edf.getSampleFrequencies()[0]))
            all_file_info[i].append(int(raw_edf.getNSamples()[0]))
            all_file_info[i].append(int(raw_edf.file_duration))

        all_file_info = pd.DataFrame(all_file_info,
                                     columns=['name', 'type', 'subject', 'session',
                                              'file', 'path', 'time', 'montage', 'freq', 'L', 'duration'])

        # Print dataset description:
        count = np.array([[all_file_info[all_file_info['type'] == 'epilepsy']['subject'].value_counts().__len__(),
                           all_file_info[all_file_info['type'] == 'no_epilepsy']['subject'].value_counts().__len__()],
                          [all_file_info[all_file_info['type'] == 'epilepsy'][
                               ['subject', 'session']].value_counts().__len__(),
                           all_file_info[all_file_info['type'] == 'no_epilepsy'][
                               ['subject', 'session']].value_counts().__len__()],
                          [all_file_info[all_file_info['type'] == 'epilepsy'][
                               ['subject', 'session', 'file']].value_counts().__len__(),
                           all_file_info[all_file_info['type'] == 'no_epilepsy'][
                               ['subject', 'session', 'file']].value_counts().__len__()]])

        print("|----------------------------------------------------------------------|")
        print("| Description |     Epilepsy     |    No Epilepsy   |      Total       |")
        print("|-------------+------------------+------------------+------------------|")
        for number, item in zip(count, ['Patients', 'Sessions', 'Files']):
            print(f"| {item:<11} | {number[0]:^16} | {number[1]:^16} | {sum(number):^16} |")
            print("|-------------+------------------+------------------+------------------|")

        length = np.array([all_file_info[all_file_info['type'] == 'epilepsy']['duration'].sum(),
                           all_file_info[all_file_info['type'] == 'no_epilepsy']['duration'].sum(),
                           all_file_info['duration'].sum()])
        print(
            f"| {'Duration':<11} | {length[0] / 3600:>7.2f}h{length[0] / length[2] * 100:>7.2f}% | {length[1] / 3600:>7.2f}h{length[1] / length[2] * 100:>7.2f}% | {length[2] / 3600:>15.2f}h |")
        print("|-------------+------------------+------------------+------------------|")

        self.restore_file_info(all_file_info)

        return all_file_list, all_file_info

    def _get_file_list(self, type):
        folder = '{}/{}'.format(self.root_path, f'{type}_edf')
        subject_list = [subject for subject in os.listdir(folder)]

        FileInfo = []
        FilePath = []
        FileList = []
        for subject in subject_list:
            subject_path = os.path.join(folder, subject)
            for session in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session)
                for session_folder in os.listdir(session_path):
                    montage = session_folder[7:]
                    files_path = os.path.join(session_path, session_folder)
                    for file in os.listdir(files_path):
                        file_path = os.path.join(files_path, file)
                        FilePath.append(file_path)
                        FileList.append(file[:-4])
                        file_info = [file[:-4],
                                     type,
                                     subject,
                                     session.split('_')[0],
                                     file[:-4].split('_')[-1],
                                     file_path,
                                     session[4:],
                                     montage]
                        FileInfo.append(file_info)

        return FileList, FilePath, FileInfo


class ReaderTUEV(Reader):

    def __init__(self, path, info_saver, montage, target_freq):

        super(ReaderTUEV, self).__init__(path, info_saver, montage, target_freq)
        if os.path.exists(self.info_saver):
            self.file_list, self.file_info = self.load_info_from_saver()
        else:
            self.file_list, self.file_info = self.get_all_file_info()

    def __getitem__(self, id):
        eeg_item = self.load_data(id)
        label = self.file_info.loc[id, 'type']
        folder = self.file_info.loc[id, 'sub_folder']
        return eeg_item, label, folder

    def load_data(self, id):
        edf_reader = EDFReader(file_path=self.file_info.loc[id, 'path'],
                               montage=self.montage['ar'],
                               target_freq=self.target_freq)
        return edf_reader

    def get_all_file_info(self):

        all_file_list = []
        all_file_path = []
        all_file_info = []
        for i, sub_folder in enumerate(['train', 'eval']):
            file_list, file_path, file_info = self._get_file_list(sub_folder)
            all_file_list += file_list
            all_file_path += file_path
            all_file_info += file_info

        for i, path in enumerate(all_file_path):

            error_file_list = []
            try:
                # try to load the raw EDF file
                '''
                There are 7 EDF files with loading errors:
                    
                    the file is not EDF(+) or BDF(+) compliant, the startdate is incorrect, 
                    it might contain incorrect characters, such as ':' instead of '.'
                    
                We need to open these file with EDFbrowser and resave the file by Header Editor tools.
                Then they can be loaded by pyedflib correctly.
                
                These 7 file path list is as following:
                
                    Error 102 in [/data2/EEGdata/TUEV/edf/train/00001057/00001057_00000001.edf]
                    Error 103 in [/data2/EEGdata/TUEV/edf/train/00001057/00001057_00000002.edf]
                    Error 104 in [/data2/EEGdata/TUEV/edf/train/00001057/00001057_00000004.edf]
                    Error 106 in [/data2/EEGdata/TUEV/edf/train/00001057/00001057_00000011.edf]
                    Error 195 in [/data2/EEGdata/TUEV/edf/train/00002047/00002047_00000005.edf]
                    Error 294 in [/data2/EEGdata/TUEV/edf/train/00002996/00002996_00000001.edf]
                    Error 323 in [/data2/EEGdata/TUEV/edf/train/00003348/00003348_00000008.edf]
                '''
                raw_edf = pyedf.EdfReader(path)
                all_file_info[i].append(int(raw_edf.getSampleFrequencies()[0]))
                all_file_info[i].append(int(raw_edf.getNSamples()[0]))
                all_file_info[i].append(int(raw_edf.file_duration))

            except Exception:
                # 捕获异常并记录错误信息
                error_file_list.append(path.split('/')[-1])
                print(f'Error {i} in [{path}]')

        all_file_info = pd.DataFrame(all_file_info,
                                     columns=['name', 'sub_folder', 'type', 'subject', 'session',
                                              'file', 'path', 'freq', 'L', 'duration'])

        self.restore_file_info(all_file_info)

        return all_file_list, all_file_info

    def _get_file_list(self, sub_folder):

        folder = '{}/{}'.format(self.root_path, f'edf/{sub_folder}')
        FileList, FilePath = find_files_with_extension(folder, extension='.edf')

        FileInfo = []

        for file, file_path in zip(FileList, FilePath):

            label = None

            if sub_folder == 'train':
                subject = file.split('_')[0]
                session = file.split('_')[0]

            if sub_folder == 'eval':

                label, subject, _, session = file.split('_')
                if session == '':
                    session = '0'

            file_info = [file,
                         sub_folder,
                         label,
                         subject,
                         session,
                         session,
                         file_path]
            FileInfo.append(file_info)

        return FileList, FilePath, FileInfo


class ReaderTUSZ(Reader):

    def __init__(self, path, info_saver, montage, target_freq):

        super(ReaderTUSZ, self).__init__(path, info_saver, montage, target_freq)
        self.event_types = []
        if os.path.exists(self.info_saver):
            self.file_list, self.file_info = self.load_info_from_saver()
        else:
            self.file_list, self.file_info = self.get_all_file_info()


    def __getitem__(self, id):
        eeg_item = self.load_data(id)
        folder = self.file_info.loc[id, 'sub_folder']
        return eeg_item, None, folder

    def load_data(self, id):
        edf_reader = EDFReader(file_path=self.file_info.loc[id, 'path'],
                               montage=self.montage[self.file_info.loc[id, 'montage']],
                               target_freq=self.target_freq)
        return edf_reader

    def load_term_label(self, path, freq):

        label_info = {}
        label = pd.read_csv(path, skiprows=5)

        if len(label) <= 0:
            return label_info

        else:
            label_with_index = (label[['start_time', 'stop_time']] * freq).round().apply(np.int64)
            label_info['num_events'] = len(label_with_index)
            label_info['num_bckg'] = len(label[label['label'] == 'bckg'])
            label_info['index_list'] = ','.join([str(x) for x in label_with_index.values.reshape(-1)])
            label_info['label_list'] = ','.join(label['label'].values)
            label_info['events_length'] = (label['stop_time'] - label['start_time']).sum()
            label_type = label['label'].value_counts().index.values
            label_info['label_type'] = ','.join(label_type)
            for name in label_type:
                if name not in self.event_types:
                    self.event_types.append(name)

            label_info['with_seizure'] = 0 if label_info['label_type'] == 'bckg' else 1

            return label_info


    def get_all_file_info(self):

        all_file_list = []
        all_file_path = []
        all_file_info = []
        for i, sub_folder in enumerate(['train', 'eval', 'dev']):
            file_list, file_path, file_info = self._get_file_list(sub_folder)
            all_file_list += file_list
            all_file_path += file_path
            all_file_info += file_info

        with tqdm(total=len(all_file_path)) as pbar:

            pbar.set_description('Load File Info of TUSZ')

            for i, path in enumerate(all_file_path):

                error_file_list = []
                try:
                    # try to load the raw EDF file
                    raw_edf = pyedf.EdfReader(path)


                except Exception:
                    # 捕获异常并记录错误信息
                    error_file_list.append(path.split('/')[-1])
                    print(f'Error {i} in [{path}]')

                all_file_info[i].append(int(raw_edf.getSampleFrequencies()[0]))
                all_file_info[i].append(int(raw_edf.getNSamples()[0]))
                all_file_info[i].append(int(raw_edf.file_duration))

                term_label_path = path[:-4] + '.csv_bi'
                label_info = self.load_term_label(term_label_path, freq=all_file_info[i][8])

                all_file_info[i].append(label_info['label_list'])
                all_file_info[i].append(label_info['index_list'])
                all_file_info[i].append(label_info['num_events'])
                all_file_info[i].append(label_info['num_bckg'])
                all_file_info[i].append(label_info['events_length'])
                all_file_info[i].append(label_info['label_type'])
                all_file_info[i].append(label_info['with_seizure'])

                pbar.update(1)

        all_file_info = pd.DataFrame(all_file_info,
                                     columns=['name', 'sub_folder', 'subject', 'session',
                                              'file', 'path', 'time', 'montage', 'freq', 'L', 'duration',
                                              'label_list', 'index_list', 'num_events', 'num_bckg', 'events_length',
                                              'label_type', 'with_seizure'])

        self.restore_file_info(all_file_info)

        return all_file_list, all_file_info

    def _get_file_list(self, sub_folder):

        folder = '{}/{}'.format(self.root_path, f'edf/{sub_folder}')
        FileList, FilePath = find_files_with_extension(folder, extension='.edf')

        FileInfo = []

        for file, file_path in zip(FileList, FilePath):

            subject, session, file_id = file.split('_')

            file_info = [file,
                         sub_folder,
                         subject,
                         session,
                         file_id,
                         file_path,
                         file_path.split('/')[-3][5:],
                         file_path.split('/')[-2][7:]]
            FileInfo.append(file_info)

        return FileList, FilePath, FileInfo


class ReaderTUSL(Reader):

    def __init__(self, path, info_saver, montage, target_freq, eval_ratio=0.1):

        super(ReaderTUSL, self).__init__(path, info_saver, montage, target_freq)

        '''
        These two raw edf file named as an illegal name:
         
        1:   .../TUSL/edf/aaaaamoa/s003_2012_10_22/01_tcp_ar/00008476_s003_t010.edf
                          aaaaamoa ------------------------> 00008476
        2:   .../TUSL/edf/aaaaanrc/s004_2012_10_11/01_tcp_ar/00009232_s004_t010.edf
                          aaaaanrc ------------------------> 00009232
        
        You'd better rename these two raw edf files manually for dataset loading.
        
        Such as:
              .../TUSL/edf/aaaaamoa/s003_2012_10_22/01_tcp_ar/aaaaamoa_s003_t010.edf
                                                          .../aaaaamoa_s003_t010.lbl_agg
                                                          .../aaaaamoa_s003_t010.tse_agg
                                                          .../aaaaamoa_s003_t010_00.lbl
                                                          .../aaaaamoa_s003_t010_00.tse
            
        '''

        self.eval_ratio = eval_ratio
        self.event_types = []
        if os.path.exists(self.info_saver):
            self.file_list, self.file_info = self.load_info_from_saver()
        else:
            self.file_list, self.file_info = self.get_all_file_info()
        if 'sub_folder' not in self.file_info.columns:
            self.prepare_train_and_eval()

    def __getitem__(self, id):
        eeg_item = self.load_data(id)
        folder = self.file_info.loc[id, 'sub_folder']
        label = self.load_label(id)
        return eeg_item, label, folder

    def load_label(self, id):
        label = pd.read_csv(self.file_info.loc[id, 'label_path'], skiprows=2, sep=' ', names=['start_time', 'stop_time', 'label', 'confidence'])
        label[['start_time', 'stop_time']] = (label[['start_time', 'stop_time']] * self.file_info.loc[id, 'freq']).round().apply(np.int64)
        label.drop(['confidence'], inplace=True, axis=1)
        return label


    def load_data(self, id):
        edf_reader = EDFReader(file_path=self.file_info.loc[id, 'path'],
                               montage=self.montage[self.file_info.loc[id, 'montage']],
                               target_freq=self.target_freq)
        return edf_reader

    def prepare_train_and_eval(self):
        subject_list = list(self.file_info['subject'].value_counts().index)

        random.seed(42)
        random.shuffle(subject_list)

        split_index = int(len(subject_list) * (1 - self.eval_ratio))
        train_list = subject_list[:split_index]
        eval_list = subject_list[split_index:]

        def assign_group(row):
            if row['subject'] in train_list:
                return 'train'
            elif row['subject'] in eval_list:
                return 'eval'
            else:
                return 'Unknown'

        self.file_info['sub_folder'] = self.file_info.apply(assign_group, axis=1)
        new_index = 1
        label_column = self.file_info.pop('sub_folder')
        self.file_info.insert(new_index, 'sub_folder', label_column)

        self.restore_file_info(self.file_info)

        return None

    def get_all_file_info(self):

        all_file_list, all_file_path, all_file_info = self._get_file_list()

        with tqdm(total=len(all_file_path)) as pbar:

            pbar.set_description('Load File Info of TUSL')

            for i, path in enumerate(all_file_path):

                error_file_list = []
                try:
                    # try to load the raw EDF file
                    raw_edf = pyedf.EdfReader(path)


                except Exception:
                    # 捕获异常并记录错误信息
                    error_file_list.append(path.split('/')[-1])
                    print(f'Error {i} in [{path}]')

                all_file_info[i].append(int(raw_edf.getSampleFrequencies()[0]))
                all_file_info[i].append(int(raw_edf.getNSamples()[0]))
                all_file_info[i].append(int(raw_edf.file_duration))

                term_label_path = path[:-4] + '.tse_agg'
                label_info = self.load_term_label(term_label_path, freq=all_file_info[i][7])

                all_file_info[i].append(term_label_path)
                all_file_info[i].append(label_info['label_list'])
                all_file_info[i].append(label_info['index_list'])
                all_file_info[i].append(label_info['num_events'])
                all_file_info[i].append(label_info['num_bckg'])
                all_file_info[i].append(label_info['events_length'])
                all_file_info[i].append(label_info['label_type'])

                pbar.update(1)

        all_file_info = pd.DataFrame(all_file_info,
                                     columns=['name', 'subject', 'session',
                                              'file', 'path', 'time', 'montage', 'freq', 'L', 'duration', 'label_path',
                                              'label_list', 'index_list', 'num_events', 'num_bckg', 'events_length',
                                              'label_type'])

        self.restore_file_info(all_file_info)

        return all_file_list, all_file_info

    def load_term_label(self, path, freq):

        label_info = {}
        label = pd.read_csv(path, skiprows=2, sep=' ', names=['start_time', 'stop_time', 'label', 'confidence'])

        if len(label) <= 0:
            return label_info

        else:
            label_with_index = (label[['start_time', 'stop_time']] * freq).round().apply(np.int64)
            label_info['num_events'] = len(label_with_index)
            label_info['num_bckg'] = len(label[label['label'] == 'bckg'])
            label_info['index_list'] = ','.join([str(x) for x in label_with_index.values.reshape(-1)])
            label_info['label_list'] = ','.join(label['label'].values)
            label_info['events_length'] = (label['stop_time'] - label['start_time']).sum()
            label_type = label['label'].value_counts().index.values
            label_info['label_type'] = ','.join(label_type)
            for name in label_type:
                if name not in self.event_types:
                    self.event_types.append(name)

            # label_info['with_seizure'] = 0 if label_info['label_type'] == 'bckg' else 1

            return label_info

    def _get_file_list(self):

        folder = '{}/{}'.format(self.root_path, 'edf')
        FileList, FilePath = find_files_with_extension(folder, extension='.edf')

        FileInfo = []

        for file, file_path in zip(FileList, FilePath):
            subject, session, file_id = file.split('_')


            file_info = [file,
                         subject,
                         session,
                         file_id,
                         file_path,
                         file_path.split('/')[-3][5:],
                         file_path.split('/')[-2][7:]]
            FileInfo.append(file_info)

        return FileList, FilePath, FileInfo

