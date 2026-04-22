from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PTBXLDataset(Dataset):
    """PTB-XL ECG Dataset loader.
    
    Beklenen dizin yapısı:
        root/
            records100/   veya records500/
            ptbxl_database.csv
            scp_statements.csv
    
    Download: https://physionet.org/content/ptb-xl/1.0.3/
    pip install wfdb
    """

    # 5 süper-sınıf
    SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "val" | "test"
        sampling_rate: int = 100,      # 100 veya 500 Hz
        window_size: int = 128,        # kaç timestamp per sample
        normalize: bool = True,
    ):
        super().__init__()
        self.root          = root
        self.split         = split
        self.sampling_rate = sampling_rate
        self.window_size   = window_size
        self.normalize     = normalize

        self.data, self.labels = self._load()

    def _load(self):
        try:
            import pandas as pd
            import wfdb
        except ImportError:
            raise ImportError("pip install wfdb pandas")

        df = pd.read_csv(os.path.join(self.root, "ptbxl_database.csv"), index_col="ecg_id")
        import ast
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        # scp_statements'ten superclass mapping yükle
        scp = pd.read_csv(os.path.join(self.root, "scp_statements.csv"), index_col=0)
        scp = scp[scp.rhythm == 1.0] if "rhythm" in scp.columns else scp

        def get_label(codes):
            for code in codes:
                if code in scp.index:
                    sc = scp.loc[code, "diagnostic_superclass"] \
                        if "diagnostic_superclass" in scp.columns else None
                    if sc in self.SUPERCLASSES:
                        return self.SUPERCLASSES.index(sc)
            return -1

        df["label"] = df["scp_codes"].apply(get_label)
        df = df[df["label"] >= 0]

        # split: strat_fold 1-8 train, 9 val, 10 test
        if self.split == "train":
            df = df[df["strat_fold"] <= 8]
        elif self.split == "val":
            df = df[df["strat_fold"] == 9]
        else:
            df = df[df["strat_fold"] == 10]

        folder = f"records{self.sampling_rate}"
        data, labels = [], []

        for ecg_id, row in df.iterrows():
            path = os.path.join(self.root, folder, row["filename_lr"]
                                if self.sampling_rate == 100 else row["filename_hr"])
            try:
                record = wfdb.rdrecord(path)
                signal = record.p_signal.astype(np.float32)  # [T, 12]
            except Exception:
                continue

            # window: ilk window_size timestamp al
            if signal.shape[0] >= self.window_size:
                signal = signal[:self.window_size]
            else:
                pad = np.zeros((self.window_size - signal.shape[0], 12), dtype=np.float32)
                signal = np.concatenate([signal, pad], axis=0)

            if self.normalize:
                mean = signal.mean(axis=0, keepdims=True)
                std  = signal.std(axis=0, keepdims=True) + 1e-8
                signal = (signal - mean) / std

            data.append(signal)
            labels.append(row["label"])

        return np.stack(data), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]),    # [window_size, 12]
            torch.tensor(self.labels[idx]),
        )


def get_ecg_loaders(
    root: str,
    batch_size: int = 32,
    window_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = PTBXLDataset(root, "train", window_size=window_size)
    val   = PTBXLDataset(root, "val",   window_size=window_size)
    test  = PTBXLDataset(root, "test",  window_size=window_size)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
