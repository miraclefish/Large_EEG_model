import os.path

import numpy as np
import pandas as pd

from DatasetReader_v1 import *
from Config import *
import pickle
import gzip
import time
from tqdm import tqdm

def save_test(data):

    data.to_csv('test.csv')

    np.savez('test.npz', data=data.values)

    with open('test.pkl', 'wb') as f:
        pickle.dump({'data':data.values}, f)

    np.savez_compressed('test_compress', data=data.values)

    with gzip.open('test_compress.pkl', 'wb') as file:
        pickle.dump({'data':data.values}, file)

    for postfix in ['.csv', '.npz', '.pkl', '_compress.npz', '_compress.pkl']:
        size = os.path.getsize('test'+postfix) / 1024 / 1024
        print(postfix, '\t: {:.2f}MB'.format(size))

    return None

def load_test():

    t = time.time()
    data1 = pd.read_csv('test.csv', index_col=0)
    t1 = time.time() - t

    t = time.time()
    data2 = np.load('test.npz')
    t2 = time.time() - t

    t = time.time()
    with open('test.pkl', 'rb') as f:
        data3 = pickle.load(f)
    t3 = time.time() - t

    t = time.time()
    data4 = np.load('test_compress.npz')
    t4 = time.time() - t

    t = time.time()
    with gzip.open('test_compress.pkl', 'rb') as file:
        data5 = pickle.load(file)
    t5 = time.time() - t

    for postfix, t in zip(['.csv', '.npz', '.pkl', '_compress.npz', '_compress.pkl'], [t1, t2, t3, t4, t5]):
        print(postfix, '\t: {:.2f}s'.format(t))

    l, c = 10, 1

    print('.csv\t\t:', data1.values[:l, c])
    print('.npz\t\t:', data2['data'][:l, c])
    print('.pkl\t\t:', data3['data'][:l, c])
    print('_compress.npz\t:', data4['data'][:l, c])
    print('_compress.pkl\t:', data5['data'][:l, c])

    return None

class DataArranger(object):

    def __init__(self, target_freq, sample_length, overlap, save_file=True, try_run=False):

        self.target_freq = target_freq
        self.sample_length = sample_length
        self.overlap = overlap
        self.dataset_name = None
        self.save_file = save_file
        self.try_run = try_run
        self.root_path = os.path.join(RAW_DATA_ROOT_PATH, DATA_SET_NAME)
        self.info_saver_path = os.path.join(WORKSPACE_PATH, DATA_INFO_SAVER)

        self.montage = MontageBuilder(path=MONTAGE_PATH)
        self.montage_dict = self.montage.initial_montage()

    def arrangeData(self, dataset):

        self.dataset_name = dataset

        arranger_dict = {
            'TUAB': self.arrangeTUAB,
            'TUEP': self.arrangeTUEP,
            'TUEV': self.arrangeTUEV,
            'TUSZ': self.arrangeTUSZ,
            'TUSL': self.arrangeTUSL,
        }

        selected_arranger = arranger_dict.get(dataset)
        selected_arranger()

        return None

    def arrangeTUAB(self):

        reader = ReaderTUAB(path=self.root_path,
                            info_saver=self.info_saver_path,
                            montage=self.montage_dict,
                            target_freq=self.target_freq)

        self.arrange_template_segment_label(reader, save_file=self.save_file)

        return None

    def arrangeTUEP(self):

        reader = ReaderTUEP(path=self.root_path,
                            info_saver=self.info_saver_path,
                            montage=self.montage_dict,
                            target_freq=self.target_freq)

        self.arrange_template_segment_label(reader, save_file=self.save_file)

        return None

    def arrangeTUEV(self):

        reader = ReaderTUEV(path=self.root_path,
                            info_saver=self.info_saver_path,
                            montage=self.montage_dict,
                            target_freq=self.target_freq)

        self.arrange_template_event_label(reader, save_file=self.save_file)

        return None

    def arrangeTUSZ(self):

        reader = ReaderTUSZ(path=self.root_path,
                            info_saver=self.info_saver_path,
                            montage=self.montage_dict,
                            target_freq=self.target_freq)

        self.arrange_template_event_label(reader, save_file=self.save_file)

        return None

    def arrangeTUSL(self):

        reader = ReaderTUSL(path=self.root_path,
                            info_saver=self.info_saver_path,
                            montage=self.montage_dict,
                            target_freq=self.target_freq)

        self.arrange_template_segment_label(reader, save_file=self.save_file)
        return None

    def arrange_template_segment_label(self, reader, save_file=True):

        dataset_sample_list = []

        n = self.get_file_number(reader)

        for i in range(n):
            eeg_data, label, folder = reader[i]

            data = eeg_data.processed_eeg.values

            if self.dataset_name == 'TUSL':
                save_folder = os.path.join(self.root_path, 'PureData/' + folder)
            else:
                save_folder = os.path.join(self.root_path, 'PureData/' + folder + '/' + label)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            L = data.shape[0]
            stride = int(self.sample_length * (1 - self.overlap))

            start_index = np.arange(0, L - self.sample_length, stride)
            end_index = np.arange(self.sample_length, L, stride)

            with tqdm(total=len(start_index)) as pbar:
                pbar.set_description('File {:05d} [{}] sampling'.format(i + 1, eeg_data.file_name))
                for j, (s, e) in enumerate(zip(start_index, end_index)):

                    save_name = eeg_data.file_name[:-4] + '_{:08d}.npz'.format(j)

                    save_path = os.path.join(save_folder, save_name)

                    if save_file:
                        data_j = data[s:e, :]
                        np.savez_compressed(save_path, data=data_j)

                    if self.dataset_name == 'TUSL':
                        sub_label = label[(label['start_time'] < e) & (label['stop_time'] > s)]

                        if len(sub_label) >= 1:
                            label_type_list = []
                            for i in range(len(sub_label)):
                                ratio = (min(e, sub_label.values[i, 1]) - max(s, sub_label.values[i, 0])) / (e - s)
                                if ratio > 0.5:
                                    label_type_list.append(sub_label.values[i, 2])

                            label_type = '_'.join(label_type_list)

                        else:
                            label_type = None

                        sample_info = [i, j, eeg_data.file_name[:-4], label_type, folder, save_path]

                    else:

                        sample_info = [i, j, eeg_data.file_name[:-4], label, folder, save_path]

                    dataset_sample_list.append(sample_info)

                    pbar.update(1)

        data_sample_info = pd.DataFrame(dataset_sample_list,
                                        columns=['file_id', 'sample_id', 'file_name', 'label', 'type', 'path'])

        if save_file:
            data_sample_info.to_csv(os.path.join(WORKSPACE_PATH, f'SampleList_{self.dataset_name}.csv'))

        print('Finish!')
        return None

    def arrange_template_event_label(self, reader, save_file=True):

        dataset_sample_list = []

        n = self.get_file_number(reader)

        for i in range(n):

            eeg_data, label, folder = reader[i]

            data = eeg_data.processed_eeg.values

            label_save_folder = os.path.join(self.root_path, 'PureData/label/' + folder)
            raw_label_save_folder = os.path.join(self.root_path, 'PureData/label')
            data_save_folder = os.path.join(self.root_path, 'PureData/data/' + folder)

            if not os.path.exists(label_save_folder):
                os.makedirs(label_save_folder)
            if not os.path.exists(data_save_folder):
                os.makedirs(data_save_folder)

            label_tabel = None
            if self.dataset_name == 'TUEV':
                label_tabel = pd.read_csv(eeg_data.file_path[:-4] + '.rec', header=None)
                label_tabel.columns = ['channel', 'start', 'end', 'class']
                label_tabel = self.preprocess_label_TUEV(label_tabel)
                raw_label_save_path = os.path.join(raw_label_save_folder, eeg_data.file_name[:-4] + '.csv')
                label_tabel.to_csv(raw_label_save_path)
            if self.dataset_name == 'TUSZ':
                label_tabel = pd.read_csv(eeg_data.file_path[:-4] + '.csv', skiprows=5)
                label_tabel.columns = ['channel', 'start', 'end', 'class', 'confidence']
                label_tabel = self.preprocess_label_TUSZ(label_tabel)
                raw_label_save_path = os.path.join(raw_label_save_folder, eeg_data.file_name[:-4] + '.npz')
                np.savez_compressed(raw_label_save_path, label=label_tabel.iloc[:, :4].values)

            L = data.shape[0]
            stride = int(self.sample_length * (1 - self.overlap))

            start_index = np.arange(0, L - self.sample_length, stride)
            end_index = np.arange(self.sample_length, L, stride)

            with tqdm(total=len(start_index)) as pbar:
                pbar.set_description('File {:05d} [{}] sampling'.format(i + 1, eeg_data.file_name))
                for j, (s, e) in enumerate(zip(start_index, end_index)):

                    data_save_name = eeg_data.file_name[:-4] + '_{:08d}.npz'.format(j)
                    label_save_name = eeg_data.file_name[:-4] + '_{:08d}.npz'.format(j)

                    data_save_path = os.path.join(data_save_folder, data_save_name)
                    label_save_path = os.path.join(label_save_folder, label_save_name)

                    if save_file:
                        data_j = data[s:e, :]
                        np.savez_compressed(data_save_path, data=data_j)
                        label_j = label_tabel[(label_tabel['start'] < e) & (label_tabel['end'] > s)]

                        if self.dataset_name == 'TUEV':
                            if len(label_j) > 0:
                                np.savez_compressed(label_save_path, label=label_j.values)
                            else:
                                label_save_path = None

                        if self.dataset_name == 'TUSZ':
                            if len(label_j) >= 1:
                                label_j.loc[:, 'start'] = label_j['start'].apply(lambda x: s if x < s else x)
                                label_j.loc[:, 'end'] = label_j['end'].apply(lambda x: e if x > e else x)

                                max_length = (label_j['end'] - label_j['start']).max()
                                label = label_j.loc[(label_j['end'] - label_j['start']) == max_length, 'class'].values[0]
                            label_save_path = None

                    sample_info = [i, j, eeg_data.file_name[:-4], label, folder, data_save_path, label_save_path, raw_label_save_path]

                    dataset_sample_list.append(sample_info)

                    pbar.update(1)

        data_sample_info = pd.DataFrame(dataset_sample_list,
                                        columns=['file_id', 'sample_id', 'file_name', 'label', 'type', 'path', 'label_path', 'raw_label_path'])

        if save_file:
            data_sample_info.to_csv(os.path.join(WORKSPACE_PATH, f'SampleList_{self.dataset_name}.csv'))

        print('Finish!')
        return None

    def get_file_number(self, reader):

        if self.try_run:
            return 5
        else:
            return len(reader)

    def preprocess_label_TUEV(self, label_table):

        label_table = label_table.sort_values(['channel', 'start'])
        grouped_label = label_table.groupby(['channel', 'class'])

        new_label = []
        for name, data in grouped_label:
            index_list = data.loc[:, ['start', 'end']].values.reshape(-1)
            new_index_list = []
            for i, index in enumerate(index_list):
                if i == 0:
                    new_index_list.append(index)
                    continue
                if index == new_index_list[-1]:
                    new_index_list.pop()
                else:
                    new_index_list.append(index)
            new_index_list = np.array(new_index_list).reshape(-1, 2)
            data = pd.DataFrame(new_index_list, columns=['start', 'end'])
            data['channel'] = name[0]
            data['class'] = name[1]
            new_label.append(data.iloc[:, [2, 0, 1, 3]])
        new_label = pd.concat(new_label).sort_values(['channel', 'start']).reset_index(drop=True)
        new_label[['start', 'end']] = (new_label[['start', 'end']] * self.target_freq).apply(np.round).apply(np.int64)

        return new_label

    def preprocess_label_TUSZ(self, label_table):

        label_table.drop('confidence', axis=1, inplace=True)
        label_table['channel_id'] = label_table['channel'].apply(lambda x: CHANNEL_TO_ID[x])
        label_table['class_id'] = label_table['class'].apply(lambda x: SEIZURE_TO_ID[x])
        label_table = label_table[['channel_id', 'start', 'end', 'class_id', 'channel', 'class']]
        label_table[['start', 'end']] = (label_table[['start', 'end']] * self.target_freq).apply(np.round).apply(np.int64)
        label_table.sort_values(['start', 'channel_id'], inplace=True)
        return label_table

if __name__ == '__main__':

    arranger = DataArranger(target_freq=125, sample_length=125*5, overlap=0.1, save_file=True, try_run=False)
    arranger.arrangeData(DATA_SET_NAME)


    # PlotEEGMontage(edf_reader.raw_eeg, 0, 10, sfreq=250)



    pass