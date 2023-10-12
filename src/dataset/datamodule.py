from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from os.path import join
import torch


class SentimentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_filepath: str,
        dev_filepath: str,
        test_filepath: str,
        dataset_class: torch.utils.data.Dataset,
        batch_size: int = 16,
        datadir: str = "data/"
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_class = dataset_class
        self.train_file = join(datadir, train_filepath)
        self.dev_file = join(datadir, dev_filepath)
        self.test_file = join(datadir, test_filepath)
        
    
    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str = "fit") -> None:
        self.train_data = self.dataset_class(self.train_file)
        self.dev_data = self.dataset_class(self.dev_file)
        self.test_data = self.dataset_class(self.test_file)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dev_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size)