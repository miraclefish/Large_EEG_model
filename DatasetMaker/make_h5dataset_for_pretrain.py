from utils.h5Dataset import h5Dataset
from pathlib import Path
from utils.eegLoader import EDFReader
import pandas as pd
import numpy as np
from Config import CHANNEL_TO_ID, SEIZURE_TO_ID, EVENT_TO_ID, SLOW_TO_ID, ARTIFACT_TO_ID


class make_h5dataset():

    def __init__(self, savePath: Path, rawDataPath: Path, l_freq: float=0.1, h_freq: float=75.0, rsfreq: int=200):

        self.savePath = savePath
        self.rawDataPath = rawDataPath
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.rsfreq = rsfreq

        self.chunks = (23, rsfreq)

    def make_dataset(self, saveName: str, datasetName: str, label_map: dict):

        group = self.rawDataPath.glob('**/*.edf')

        dataset = h5Dataset(self.savePath, saveName)

        for i, edfFile in enumerate(group):

            print(f'No. [{i}] -----------------------------------------------------')
            print(f'processing {edfFile.name}')

            grp = dataset.addGroup(grpName=edfFile.stem)

            montage_type = self.get_montage_type(edfFile, datasetName)

            reader = EDFReader(edfFile, montage_type, self.l_freq, self.h_freq, self.rsfreq)
            chOrder = list(reader.processed_data.columns)
            eegData = reader.processed_data.values.T

            dset = dataset.addDataset(grp, 'eeg', eegData, self.chunks)


            # dataset attributes
            dataset.addAttributes(dset, 'lFreq', self.l_freq)
            dataset.addAttributes(dset, 'hFreq', self.h_freq)
            dataset.addAttributes(dset, 'rsFreq', self.rsfreq)
            dataset.addAttributes(dset, 'chOrder', chOrder)
            dataset.addAttributes(dset, 'setName', datasetName)
            dataset.addAttributes(dset, 'montage', montage_type)
            dataset.modifyAttributes(grpName=edfFile.stem, dsName='eeg', attrName='setName', attrValue=datasetName)
            dataset.modifyAttributes(grpName=edfFile.stem, dsName='eeg', attrName='montage', attrValue=montage_type)

            # dataset label
            if datasetName == 'TUAB':
                label = label_map[edfFile.parent.parent.name]
                dataset.addAttributes(dset, 'label', label)
            if datasetName == 'TUEP':
                label = label_map[edfFile.parent.parent.parent.parent.name]
                dataset.addAttributes(dset, 'label', label)
            if datasetName == 'TUSZ':
                labelFileChannelWise = edfFile.parent / (edfFile.stem + '.csv')
                labelChannelWise = self.parsing_event_label(labelFileChannelWise, datasetName, self.rsfreq, label_map, mode='channel')
                labelFileTermWise = edfFile.parent / (edfFile.stem + '.csv_bi')
                labelTermWise = self.parsing_event_label(labelFileTermWise, datasetName, self.rsfreq, label_map, mode='term')
                dataset.addAttributes(dset, 'labelChannel', labelChannelWise)
                dataset.addAttributes(dset, 'labelTerm', labelTermWise)
            if datasetName == 'TUEV':
                labelFileChannelWise = edfFile.parent / (edfFile.stem + '.rec')
                labelChannelWise = self.parsing_event_label(labelFileChannelWise, datasetName, self.rsfreq, label_map, mode='channel')
                dataset.addAttributes(dset, 'labelChannel', labelChannelWise)
            if datasetName == 'TUSL':
                labelFileTermWise = edfFile.parent / (edfFile.stem + '.tse_agg')
                labelTermWise = self.parsing_event_label(labelFileTermWise, datasetName, self.rsfreq, label_map, mode='term')
                dataset.addAttributes(dset, 'labelTerm', labelTermWise)
            if datasetName == 'TUAR':
                labelFileChannelWise = edfFile.parent / (edfFile.stem + '.csv')
                labelChannelWise = self.parsing_event_label(labelFileChannelWise, datasetName, self.rsfreq, label_map, mode='channel')
                dataset.addAttributes(dset, 'labelChannel', labelChannelWise)

        dataset.save()
        return None

    def get_montage_type(self, edfFile, datasetName):

        if datasetName in ['TUAB', 'TUEP', 'TUSZ', 'TUSL', 'TUAR']:
            return edfFile.parent.name[7:]
        if datasetName in ['TUEV']:
            return 'ar'

    def parsing_event_label(self, labelFile, datasetName, rsFreq, label_map, mode):

        if datasetName in ['TUSZ']:

            label_table = pd.read_csv(labelFile, skiprows=5)

            label_table['label'] = label_table['label'].apply(lambda x: label_map[x])
            label_table[['start_time', 'stop_time']] = label_table[['start_time', 'stop_time']].apply(
                lambda x: np.int64(x * rsFreq))
            label_table.drop('confidence', axis=1, inplace=True)
            label_table.sort_values(['start_time', 'stop_time'], inplace=True)

            if mode == 'channel':
                label_table['channel'] = label_table['channel'].apply(lambda x: CHANNEL_TO_ID[x])
            if mode == 'term':
                label_table.drop('channel', axis=1, inplace=True)

        if datasetName in ['TUEV']:

            label_table = pd.read_csv(labelFile, header=None)
            label_table.columns = ['channel', 'start_time', 'stop_time', 'label']
            label_table[['start_time', 'stop_time']] = label_table[['start_time', 'stop_time']].apply(
                lambda x: np.int64(x * rsFreq))
            label_table['label'] = label_table['label'].apply(lambda x: 0 if x == 6 else x)
            label_table.sort_values(['start_time', 'stop_time'], inplace=True)

        if datasetName in ['TUSL']:

            label_table = pd.read_csv(labelFile, header=None, skiprows=2, sep=' ')
            label_table.columns = ['start_time', 'stop_time', 'label', 'confidence']
            label_table[['start_time', 'stop_time']] = label_table[['start_time', 'stop_time']].apply(
                lambda x: np.int64(x * rsFreq))
            label_table.drop('confidence', axis=1, inplace=True)
            label_table['label'] = label_table['label'].apply(lambda x: label_map[x])
            label_table.sort_values(['start_time', 'stop_time'], inplace=True)

        if datasetName in ['TUAR']:

            label_table = pd.read_csv(labelFile, skiprows=6)

            label_table['label'] = label_table['label'].apply(lambda x: label_map[x])
            label_table[['start_time', 'stop_time']] = label_table[['start_time', 'stop_time']].apply(
                lambda x: np.int64(x * rsFreq))
            label_table.drop('confidence', axis=1, inplace=True)
            label_table.sort_values(['start_time', 'stop_time'], inplace=True)

            label_table['channel'] = label_table['channel'].apply(lambda x: CHANNEL_TO_ID[x])

            def multi_channel_to_string(x):
                word = []
                for i in range(22):
                    if i in x:
                        word.append('1')
                    else:
                        word.append('0')
                return int(''.join(word), 2)

            label_table = label_table.groupby(['start_time', 'stop_time']).agg(
                {'channel': lambda x: list(x), 'label': lambda x: set(x).pop()})
            label_table['channel'] = label_table['channel'].apply(multi_channel_to_string)
            label_table.reset_index(inplace=True)
            label_table = label_table[['channel', 'start_time', 'stop_time', 'label']]

        return label_table.values




if __name__ == '__main__':

    from Config import RAW_DATA_ROOT_PATH

    # # TUAB dataset making ...
    # datasetName = 'TUAB'
    # sub_folder = 'eval'
    # rootPath = Path(RAW_DATA_ROOT_PATH) / datasetName
    #
    # savePath = rootPath / 'Processed'
    # rawDataPath = rootPath / f'edf/{sub_folder}'
    # maker = make_h5dataset(savePath=savePath, rawDataPath=rawDataPath, l_freq=0.1, h_freq=75.0, rsfreq=200)
    # label_map = {'normal': 0, 'abnormal': 1}
    # maker.make_dataset(saveName=f'{datasetName}{sub_folder}', datasetName=datasetName, label_map=label_map)

    # # TUAR dataset making ...
    # datasetName = 'TUAR'
    # rootPath = Path(RAW_DATA_ROOT_PATH) / datasetName
    #
    # savePath = rootPath / 'Processed'
    # rawDataPath = rootPath / 'edf'
    # maker = make_h5dataset(savePath=savePath, rawDataPath=rawDataPath, l_freq=0.1, h_freq=75.0, rsfreq=200)
    # label_map = ARTIFACT_TO_ID
    # maker.make_dataset(saveName=f'{datasetName}', datasetName=datasetName, label_map=label_map)

    # # TUEP dataset making ...
    # datasetName = 'TUEP'
    # rootPath = Path(RAW_DATA_ROOT_PATH) / datasetName
    #
    # savePath = rootPath / 'Processed'
    # rawDataPath = rootPath
    # maker = make_h5dataset(savePath=savePath, rawDataPath=rawDataPath, l_freq=0.1, h_freq=75.0, rsfreq=200)
    # label_map = {'no_epilepsy_edf': 0, 'epilepsy_edf': 1}
    # maker.make_dataset(saveName=datasetName, datasetName=datasetName, label_map=label_map)

    # # TUEV dataset making ...
    # datasetName = 'TUEV'
    # sub_folder = 'train'
    # rootPath = Path(RAW_DATA_ROOT_PATH) / datasetName
    #
    # savePath = rootPath / 'Processed'
    # rawDataPath = rootPath / f'edf/{sub_folder}'
    # maker = make_h5dataset(savePath=savePath, rawDataPath=rawDataPath, l_freq=0.1, h_freq=75.0, rsfreq=200)
    # label_map = EVENT_TO_ID
    # maker.make_dataset(saveName=f'{datasetName}{sub_folder}', datasetName=datasetName, label_map=label_map)

    # TUSL dataset making ...
    datasetName = 'TUSL'
    rootPath = Path(RAW_DATA_ROOT_PATH) / datasetName

    savePath = rootPath / 'Processed'
    rawDataPath = rootPath / 'edf'
    maker = make_h5dataset(savePath=savePath, rawDataPath=rawDataPath, l_freq=0.1, h_freq=75.0, rsfreq=200)
    label_map = SLOW_TO_ID
    maker.make_dataset(saveName=f'{datasetName}', datasetName=datasetName, label_map=label_map)

    # # TUSZ dataset making ...
    # datasetName = 'TUSZ'
    # sub_folder = 'dev'
    # rootPath = Path(RAW_DATA_ROOT_PATH) / datasetName
    #
    # savePath = rootPath / 'Processed'
    # rawDataPath = rootPath / f'edf/{sub_folder}'
    # maker = make_h5dataset(savePath=savePath, rawDataPath=rawDataPath, l_freq=0.1, h_freq=75.0, rsfreq=200)
    # label_map = SEIZURE_TO_ID
    # maker.make_dataset(saveName=f'{datasetName}{sub_folder}', datasetName=datasetName, label_map=label_map)


    pass