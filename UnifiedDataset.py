import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


class UnifiedDataset(Dataset):
    def __init__(self, args, data_dir, split, cut=64600, return_path=False):
        self.args = args
        self.data_dir = data_dir
        self.split = split
        self.cut = cut
        self.return_path = return_path

        meta_path = os.path.join(data_dir, split, "meta.csv")
        self.meta = pd.read_csv(meta_path, dtype=str, low_memory=False).to_dict("records")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta[index]
        utt_id = row["sample_name"]
        label = int(row["is_fake"])

        audio_path = self._get_audio_path(row, utt_id, label)
        audio_path = resolve_audio_path(audio_path)

        waveform, _ = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = pad_or_truncate(waveform.astype(np.float32), self.cut)
        x = torch.from_numpy(waveform).float()
        if self.return_path:
            return x, label, audio_path
        return x, label

    def _get_audio_path(self, row, utt_id, label):
        for key in ("audio_path", "rel_path", "path"):
            if key not in row or pd.isna(row[key]):
                continue

            path_value = str(row[key])
            if os.path.isabs(path_value):
                return path_value

            split_relative = os.path.join(self.data_dir, self.split, path_value)
            dataset_relative = os.path.join(self.data_dir, path_value)
            if os.path.exists(split_relative):
                return split_relative
            if os.path.exists(dataset_relative):
                return dataset_relative
            return split_relative

        return os.path.join(
            self.data_dir,
            self.split,
            "fake" if label == 1 else "real",
            str(utt_id),
        )


def resolve_audio_path(path):
    path = os.path.abspath(path)
    if os.path.exists(path):
        return path

    root, ext = os.path.splitext(path)
    if ext:
        raise FileNotFoundError(path)

    for candidate_ext in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = root + candidate_ext
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Audio file not found for {path}. Tried extensions: "
        f"{', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
    )


def pad_or_truncate(x, max_len):
    if len(x) >= max_len:
        return x[:max_len]

    out = np.zeros(max_len, dtype=np.float32)
    out[: len(x)] = x
    return out
