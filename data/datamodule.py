import os
import lightning as L
from .dataset import SpeechDataset
from torch.utils.data import DataLoader

# should organize the data like this
# dataset/
#   train/
#       domain_A /
#           file1.wav
#           file2.wav
#           ...
#       domain_B/
#           file1.wav
#           file2.wav
#        ...
#   test/
#       domain_A /
#           file1.wav
#           file2.wav
#           ...
#       domain_B/
#           file1.wav
#           file2.wav
#        ...
#   val/
#       ...


class SpeechDataModule(L.LightningDataModule):
    def __init__(self, dm_config):
        super().__init__()
        self.save_hyperparameters()
        self.datasets = {}
        for split in dm_config.split:
            self.datasets[split] = SpeechDataset(
                path=os.path.join(dm_config.root_path, split),
                **dm_config.dataset,
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.hparams.dm_config.batch_size,
            shuflfe=True,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
            persistent_workers=self.hparams.dm_config.persistent_workers,
            # collate_fn=self.collate_fn,
        )

    def valid_dataloader(self):
        return DataLoader(
            self.datasets["valid"],
            batch_size=self.hparams.dm_config.batch_size,
            shuflfe=False,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
            persistent_workers=self.hparams.dm_config.persistent_workers,
            # collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.hparams.dm_config.batch_size,
            shuflfe=False,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
            persistent_workers=self.hparams.dm_config.persistent_workers,
            # collate_fn=self.collate_fn,
        )

    # def collate_fn(self, batch):
    #     n_features = len(batch)
    #     data = []
    #     length = []
    #     for i in batch:
    #         datum = [j for j in i]
    #         length.append([])
