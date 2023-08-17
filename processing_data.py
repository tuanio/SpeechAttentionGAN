# process all data then saving to pickle file
# loading later for saving time purpose
# training data and predicting data is different
# training data is cutting to components and then shuffle for training
# predicting data is cutting to components then save the order to synthesize to audio later
# save to pickle
# split
#   domain.pkl
# example...
# train
#   clean.pkl
#   noisy.pkl
# test
#   clean.pkl

import os
import glob
import torch
import math
import torchaudio
import argparse
from functools import partial
import torchaudio.transforms as T
import concurrent.futures as cf

N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 64

PARAMS = dict(n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH)

FIX_W = 128
audio_ext = ["flac", "wav", "mp3"]

stft = T.Spectrogram(**PARAMS, power=None)
# istft = T.InverseSpectrogram(**PARAMS)

def log(text):
    os.system(f'echo {text} >> temp_log.txt')

def cutting(img, is_train: bool = True, fix_w: int = FIX_W):
    max_size = img.size(-1)
    l = []
    curr_idx = 0
    while img.size(-1) - curr_idx > fix_w:
        l.append(img[:, :, curr_idx : curr_idx + fix_w])
        curr_idx += fix_w
    remain = max_size - curr_idx
    if remain and is_train:
        add = fix_w - remain
        roll_img = torch.tile(img, (math.ceil(add / fix_w),))
        remain_tensor = img[:, :, curr_idx:]
        add_on_tensor = roll_img[:, :, :add]
        l.append(torch.cat([remain_tensor, add_on_tensor], dim=-1))
    return l, max_size


def getting_all_audio(path):
    data = []
    for ext in audio_ext:
        data.extend(glob.glob(path + "/*." + ext))
    return data


def processing(pack, is_train):
    idx, path = pack
    log(idx)
    wav, sr = torchaudio.load(path)
    spectrogram = stft(wav)
    magnitude = torch.abs(spectrogram)
    phase = torch.angle(spectrogram)

    mag_coms, mag_max_size = cutting(magnitude, is_train)
    phase_coms, phase_max_size = cutting(phase, is_train)

    packs = [dict(magnitude=m, phase=p) for m, p in zip(mag_coms, phase_coms)]
    ret_data = {"data": packs, "max_size": mag_max_size, "path": path}
    return ret_data


def main(args):
    data_path = os.path.join(args.path, args.split)
    src_data_path = os.path.join(data_path, args.src_domain)
    tgt_data_path = os.path.join(data_path, args.tgt_domain)

    print("Getting audios...")
    all_src_audio_path = getting_all_audio(src_data_path)
    all_tgt_audio_path = getting_all_audio(tgt_data_path)

    print(
        f"Total audio | {args.src_domain}: {len(all_src_audio_path)} | {args.tgt_domain}: {len(all_tgt_audio_path)}"
    )

    is_train = args.stage == "train"

    root_save_path = os.path.join(args.dest_path, args.stage)
    src_save_path = os.path.join(root_save_path, args.src_domain + '.pt')
    tgt_save_path = os.path.join(root_save_path, args.tgt_domain + '.pt')

    if not os.path.exists(src_save_path):
        os.system("mkdir -p " + src_save_path)

    if not os.path.exists(tgt_save_path):
        os.system("mkdir -p " + tgt_save_path)

    log('process source')
    print(f"Processing for {args.src_domain}")
    with cf.ThreadPoolExecutor(max_workers=args.threads) as exe:
        src_data = list(
            exe.map(partial(processing, is_train=is_train), enumerate(all_src_audio_path))
        )
    print(f"Saving {args.src_domain} data.")
    torch.save(src_data, src_save_path)
    log('saved source')

    log('process target')
    print(f"Processing for {args.tgt_domain}")
    with cf.ThreadPoolExecutor(max_workers=args.threads) as exe:
        tgt_data = list(
            exe.map(partial(processing, is_train=is_train), enumerate(all_tgt_audio_path))
        )
    print(f"Saving {args.tgt_domain} data.")
    torch.save(tgt_data, tgt_save_path)
    log('saved target')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument(
        "--split", type=str, default="train"
    )  # train, test, valid, predict (name of a folder)
    parser.add_argument("--src-domain", type=str, default="clean")  # name of subfolder
    parser.add_argument("--tgt-domain", type=str, default="noisy")  # name of subfolder
    parser.add_argument("--dest-path", type=str)
    parser.add_argument(
        "--stage", type=str, default="train"
    )  # create for train, test, predict
    parser.add_argument(
        "--threads", type=int, default=8
    )  # create for train, test, predict
    args = parser.parse_args()

    main(args)
