import bisect
import h5py
from pathlib import Path
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0

        with tqdm(total=len(self.__subjects)) as pbar:
            pbar.set_description(f'Initialize dataset from {self.__file_path.name}')
            for subject in self.__subjects:
                self.__global_idxes.append(global_idx)  # the start index of the subject's sample in the dataset
                subject_len = self.__file[subject]['eeg'].shape[1]
                # total number of samples
                total_sample_num = (subject_len - self.__window_size) // self.__stride_size + 1
                # cut out part of samples
                start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size
                end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

                self.__local_idxes.append(start_idx)
                global_idx += (end_idx - start_idx) // self.__stride_size + 1
                pbar.update(1)
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        eeg_data = self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx + self.__window_size]
        return eeg_data, label

    def free(self) -> None:
        if self.__file:
            self.__file.close()
            self.__file = None

    def event_label_parsing(self, subject_idx, start_idx, stop_idx):
        label = self.__file[self.__subjects[subject_idx]]['eeg'].attrs['labelChannel']


    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']

    @property
    def subject_number(self):
        return len(self.__subjects)



class NpzDataset():

    def __init__(self, info_file, type='Train'):

        self.info_file = info_file
        self.type = type
        self.info = self.load_sample_info()
        self.path_list = list(self.info.path)
        self.label_list = list(self.info.label)
        self.label_map = {'normal': 0, 'abnormal': 1}

    def __getitem__(self, idx):
        data = np.load(self.path_list[idx])['data']
        label = self.label_map[self.label_list[idx]]
        return data, label

    def __len__(self):
        return len(self.path_list)

    def load_sample_info(self):
        info = pd.read_csv(self.info_file, index_col=0)
        info = info[info['type'] == 'train']
        return info


if __name__ == '__main__':

    file_path = Path('/root/autodl-pub/YYF/EEGdata/TUEV/Processed/TUEVeval.hdf5')
    dataset = SingleShockDataset(file_path, 1000, 900, 0, 1)
    data, label = dataset[5]


    from torch.utils.data import DataLoader
    import time
    file_path = Path('/root/autodl-pub/YYF/EEGdata/TUAB/Processed/TUABtrain.hdf5')
    dataset = SingleShockDataset(file_path, 1000, 900, 0, 1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    T = []
    t = time.time()
    for i, batch_data in enumerate(dataloader):
        if i >= 50:
            break
        data, label = batch_data
        print(f'hdf5 loader: batch {i} {data.shape}')
        T.append(time.time() - t)
        t = time.time()


    file_path = Path('/root/autodl-pub/YYF/Large_EEG_model/EEGdata/SampleList_TUAB.csv')
    dataset = NpzDataset(info_file=file_path, type='train')

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    T2 = []
    t = time.time()
    for i, batch_data in enumerate(dataloader):
        if i >= 50:
            break
        data, label = batch_data
        print(f'hdf5 loader: batch {i} {data.shape}')
        T2.append(time.time() - t)
        t = time.time()

    pass
