import h5py
import numpy as np
from pathlib import Path

class h5Dataset:
    def __init__(self, path: Path, name: str) -> None:

        if not Path.exists(path):
            Path.mkdir(path)
        self.__name = name
        self.__f = h5py.File(path / f'{name}.hdf5', 'a')

    def addGroup(self, grpName: str):
        grp = self.__f.create_group(grpName)
        return grp

    def addDataset(self, grp: h5py.Group, dsName: str, arr: np.array, chunks: tuple):
        return grp.create_dataset(dsName, data=arr, chunks=chunks)

    def addAttributes(self, src: 'h5py.Dataset|h5py.Group', attrName: str, attrValue):
        src.attrs[f'{attrName}'] = attrValue

    def modifyAttributes(self, grpName: str, dsName: str, attrName: str, attrValue):
        self.__f[grpName][dsName].attrs[f'{attrName}'] = attrValue

    def save(self):
        self.__f.close()

    @property
    def name(self):
        return self.__name

