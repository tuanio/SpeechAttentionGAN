import os
import torch
import random
import torchaudio
import torchvision.transforms as TVT
import torchvision
from typing import List
from collections import defaultdict
from torch.utils.data import Dataset

pickle_ext = ".pt"
audio_exts = ["wav", "flac", "mp3"]


# not resizing the image, but work with (129, 128)
class SpeechDataset(Dataset):
    '''
        this one work for 
    '''
    def __init__(
        self, path: str, domains: List[str], src_domain: str, is_train: bool = True
    ):
        super().__init__()

        self.domains = domains
        self.src_domain = src_domain
        self.tgt_domain = domains[0] if src_domain == domains[1] else domains[1]

        src_data = torch.load(os.path.join(path, self.src_domain + pickle_ext))
        tgt_data = torch.load(os.path.join(path, self.tgt_domain + pickle_ext))

        if is_train or not is_train:
            self.src_pool = [
                mag_and_phase for i in src_data for mag_and_phase in i["data"]
            ]
            self.tgt_pool = [
                mag_and_phase for i in tgt_data for mag_and_phase in i["data"]
            ]
        else:
            # for testing, but currently train only
            pass

        # normalize to range [-1, 1] to map with tanh activation in gan
        self.transforms = TVT.Compose([TVT.Normalize(mean=(0.5), std=(0.5))])

    def __len__(self):
        return len(self.src_pool)

    def __getitem__(self, idx):
        # return magnitude_A (input_A), magnitude_B (input_B), phase_A (input_phase_A), phase_B (input_phase_B)
        # fix select by index
        src_audio = self.src_pool[idx]
        # randomly choice tgt domain
        tgt_audio = random.choice(self.tgt_data)

        magnitude_A = self.transforms(src_audio["magnitude"])
        phase_A = self.transforms(src_audio["phase"])

        magnitude_B = self.transforms(tgt_audio["magnitude"])
        phase_A = self.transforms(tgt_audio["phase"])

        # every feature have shape (1, n_fft // 2 + 1, seq_len)

        return magnitude_A, magnitude_B, phase_A, phase_B