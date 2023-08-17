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
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        datasets = {}
        for split in cfg.split:
            datasets[split] = SpeechDataset(
                path=os.path.join(cfg.root_path, split),
                **cfg.dataset,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset["train"],
            batch_size=self.hparams.cfg.batch_size,
            shuflfe=True,
            num_workers=self.hparams.cfg.num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            persistent_workers=self.hparams.cfg.persistent_workers,
            # collate_fn=self.collate_fn,
        )

    def valid_dataloader(self):
        return DataLoader(
            dataset["valid"],
            batch_size=self.hparams.cfg.batch_size,
            shuflfe=False,
            num_workers=self.hparams.cfg.num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            persistent_workers=self.hparams.cfg.persistent_workers,
            # collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset["test"],
            batch_size=self.hparams.cfg.batch_size,
            shuflfe=False,
            num_workers=self.hparams.cfg.num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            persistent_workers=self.hparams.cfg.persistent_workers,
            # collate_fn=self.collate_fn,
        )

    # def collate_fn(self, batch):
    #     n_features = len(batch)
    #     data = []
    #     length = []
    #     for i in batch:
    #         datum = [j for j in i]
    #         length.append([])
