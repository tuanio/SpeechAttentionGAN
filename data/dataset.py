import torch
import random
import torchaudio
import torchaudio.transforms as T
import torchvision
from typing import List
from collections import defaultdict
from torch.utils.data import Dataset

audio_exts = ["wav", "flac", "mp3"]


# not resizing the image, but work with (129, 128)
class SpeechDataset(Dataset):
    def __init__(
        self, path: str, domains: List[str], src_domain: str, stft_params: dict
    ):
        super().__init__()

        self.data_pool = defaultdict[list]
        for domain in domains:
            self.data_pool[domain] = self.get_all_audio_paths(
                os.path.join(path, domain)
            )
        self.domains = domains
        self.src_domain = src_domain
        self.tgt_domain = domains[0] if src_domain == domains[1] else domains[1]
        self.stft_params = stft_params
        self.stft = T.Spectrogram(power=None, **stft_params)

    def get_all_audio_paths(self, path):
        datasets = []
        for ext in audio_exts:
            datasets.extend(glob.glob(path + os.sep + "*." + ext))

    def extract_magphase(self, wav):
        spectrogram = self.stft(wav)
        magnitude = torch.abs(spectrogram)
        phase = torch.angle(spectrogram)
        return magnitude, phase

    def __len__(self):
        return len(self.data_pool[self.src_domain])

    def __getitem__(self, idx):
        # return magnitude_A (input_A), magnitude_B (input_B), phase_A (input_phase_A), phase_B (input_phase_B)
        # fix select by index
        src_audio = self.data_pool[self.src_domain][idx]
        # randomly choice tgt domain
        tgt_audio = random.choice(self.data_pool[self.tgt_domain])

        wav_src, sr_src = torchaudio.load(src_audio)
        wav_tgt, sr_tgt = torchaudio.load(tgt_audio)

        magnitude_A, phase_A = self.extract_magphase(wav_src)
        magnitude_B, phase_B = self.extract_magphase(wav_tgt)

        list_mag_A, mag_A_max_size = cutting(magnitude_A)
        list_phase_A, phase_A_max_size = cutting(phase_A)
        list_mag_B, mag_B_max_size = cutting(magnitude_B)
        list_phase_B, phase_B_max_size = cutting(phase_B)

        # every feature have shape (1, n_fft // 2 + 1, seq_len)

        return magnitude_A, magnitude_B, phase_A, phase_B
