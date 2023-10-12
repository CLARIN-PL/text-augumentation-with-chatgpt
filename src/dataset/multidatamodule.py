from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from os.path import join
import torch
from typing import List


class MultiFileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_filepaths: List[str],
        dev_filepath: str,
        test_filepath: str,
        dataset_class: torch.utils.data.Dataset,
        new_train_filepath: str,
        batch_size: int = 16,
        datadir: str = "data/",
        new_proportions: float = 1.0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_class = dataset_class
        self.train_files = [join(datadir, filepath) for filepath in train_filepaths]
        self.dev_file = join(datadir, dev_filepath)
        self.test_file = join(datadir, test_filepath)
        
        self.new_train_file = join(datadir, new_train_filepath)
        self.new_proportions = new_proportions
    
    def prepare_data(self) -> None:
        curr_data = {"DOCUMENT": [], "TRUE_SENTIMENT": []}
        for filepath in self.train_files:
            curr_dataset = self.dataset_class(filepath)
            curr_X, curr_Y = curr_dataset.X, curr_dataset.Y
            if not isinstance(curr_X, list):
                curr_X = curr_X.tolist()
            if not isinstance(curr_Y, list):
                curr_Y = curr_Y.tolist()
            curr_data["DOCUMENT"].extend(curr_X)
            curr_data["TRUE_SENTIMENT"].extend(curr_Y)
        df = pd.DataFrame(curr_data)
        df.to_csv(self.new_train_file, index=False)

    def setup(self, stage: str = "fit") -> None:
        self.train_data = self.dataset_class(self.new_train_file)
        self.dev_data = self.dataset_class(self.dev_file)
        self.test_data = self.dataset_class(self.test_file)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dev_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size)