from __future__ import annotations

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _check_ecg_failure_rate(failed: list, total: int) -> None:
    """Raise RuntimeError if the ECG failure rate exceeds 10%."""
    if total > 0 and len(failed) / total > 0.10:
        raise RuntimeError(
            f"ECG loading failure rate {len(failed)}/{total} exceeds 10%. "
            f"First failed IDs: {failed[:5]}"
        )


def _fit_window(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Trim or right-pad a [T, 12] signal to exactly ``window_size`` timesteps.

    The default window is the *full* record (1000 samples = 10 s at 100 Hz), so
    this is normally a no-op: truncating to a short prefix would discard most of
    the ECG and unfairly handicap the model relative to baselines like
    xresnet1d101, which consume the whole 1000-sample signal.
    """
    t = signal.shape[0]
    if t >= window_size:
        return signal[:window_size]
    pad = np.zeros((window_size - t, signal.shape[1]), dtype=signal.dtype)
    return np.concatenate([signal, pad], axis=0)


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
        split: str = "train",  # "train" | "val" | "test"
        sampling_rate: int = 100,  # 100 veya 500 Hz
        window_size: int = 1000,  # full 10 s record at 100 Hz (use 5000 for 500 Hz)
        normalize: bool = True,
        multilabel: bool = False,  # multi-hot superclass targets (PTB-XL protocol)
        task: str = "superdiag",  # superdiag | subdiag | diag | form | rhythm | all
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.normalize = normalize
        self.task = task
        # Any task other than the legacy single-label super-diagnostic is multi-label.
        self.multilabel = multilabel or task != "superdiag"

        self.classes: list[str] = list(self.SUPERCLASSES)  # set in _load from the vocab
        self.num_classes: int = len(self.classes)
        self.data, self.labels = self._load()

    def _load(self):
        try:
            import pandas as pd
            import wfdb
        except ImportError:
            raise ImportError("pip install wfdb pandas")

        from prism.data.ptbxl_tasks import record_labels, task_vocab

        df = pd.read_csv(os.path.join(self.root, "ptbxl_database.csv"), index_col="ecg_id")
        import ast

        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        # Full scp_statements table (NOT pre-filtered — task_vocab/record_labels
        # filter per task). Plain-dict view keeps the mapping logic pandas-free.
        scp_df = pd.read_csv(os.path.join(self.root, "scp_statements.csv"), index_col=0)
        scp = scp_df.to_dict("index")

        # Split-independent label vocabulary for the task.
        self.classes = task_vocab(scp, self.task)
        self.num_classes = len(self.classes)
        class_index = {c: i for i, c in enumerate(self.classes)}

        df["labelnames"] = df["scp_codes"].apply(lambda c: record_labels(scp, c, self.task))
        df = df[df["labelnames"].apply(len) > 0]  # keep records with ≥1 label

        # split: strat_fold 1-8 train, 9 val, 10 test
        if self.split == "train":
            df = df[df["strat_fold"] <= 8]
        elif self.split == "val":
            df = df[df["strat_fold"] == 9]
        else:
            df = df[df["strat_fold"] == 10]

        data, labels = [], []
        failed: list = []

        for ecg_id, row in df.iterrows():
            path = os.path.join(
                self.root,
                row["filename_lr"] if self.sampling_rate == 100 else row["filename_hr"],
            )
            try:
                record = wfdb.rdrecord(path)
                signal = record.p_signal.astype(np.float32)  # [T, 12]
            except Exception as e:
                logger.warning("Failed to load ECG record %s from %s: %s", ecg_id, path, e)
                failed.append(ecg_id)
                continue

            signal = _fit_window(signal, self.window_size)

            if self.normalize:
                mean = signal.mean(axis=0, keepdims=True)
                std = signal.std(axis=0, keepdims=True) + 1e-8
                signal = (signal - mean) / std

            data.append(signal)
            labels.append(row["labelnames"])  # list[str] of class names for this record

        total = len(df)
        loaded = total - len(failed)
        logger.info(
            "ECG split=%s task=%s classes=%d: loaded %d/%d records (%d failed)",
            self.split, self.task, self.num_classes, loaded, total, len(failed),
        )
        _check_ecg_failure_rate(failed, total)

        if self.multilabel:
            multihot = np.zeros((len(labels), self.num_classes), dtype=np.float32)
            for i, names in enumerate(labels):
                for name in names:
                    multihot[i, class_index[name]] = 1.0
            return np.stack(data), multihot
        # legacy single-label super-diagnostic: first class name → its index
        idxs = np.array([class_index[names[0]] for names in labels], dtype=np.int64)
        return np.stack(data), idxs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # multi-hot float vector (multilabel) or scalar long (single-label)
        label_t = torch.from_numpy(label) if self.multilabel else torch.tensor(label)
        return (
            torch.from_numpy(self.data[idx]),  # [window_size, 12]
            label_t,
        )


def get_ecg_loaders(
    root: str,
    batch_size: int = 32,
    window_size: int = 128,
    num_workers: int = 4,
    multilabel: bool = False,
    task: str = "superdiag",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kw = dict(window_size=window_size, multilabel=multilabel, task=task)
    train = PTBXLDataset(root, "train", **kw)
    val = PTBXLDataset(root, "val", **kw)
    test = PTBXLDataset(root, "test", **kw)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
