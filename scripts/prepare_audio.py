"""Download Speech Commands v0.02 and build the mel-patch dumps that
``prism/data/audio.get_audio_loaders(synthetic=False)`` expects.

Codec-free path (NixOS-friendly): the tarball is fetched with urllib, wav
files are decoded with the stdlib ``wave`` module (16-bit PCM @ 16 kHz),
and the mel spectrogram uses torchaudio's pure-torch transform (no
TorchCodec / ffmpeg dependency).

Output: TensorDataset(feats[num_frames=128, mel=64], labels) at
``<root>/train.pt`` and ``<root>/val.pt`` (official validation/testing
split lists), limited to the 10 core commands.

Usage:
    python scripts/prepare_audio.py --root datasets/audio [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

import torch

COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.bz2"
NUM_FRAMES = 128
MEL_BINS = 64
SAMPLE_RATE = 16000


def _download_and_extract(root: Path) -> Path:
    tar_path = root / "speech_commands_v0.02.tar.bz2"
    out_dir = root / "speech_commands_v0.02"
    if out_dir.is_dir():
        return out_dir
    if not tar_path.is_file():
        print(f"downloading {URL} (~2.3 GB) …", flush=True)
        urllib.request.urlretrieve(URL, tar_path)
    print("extracting …", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(root)
    return out_dir


def _read_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return torch.frombuffer(frames, dtype=torch.int16).float() / 32768.0


def _split_sets(sc_root: Path) -> tuple[set, set]:
    val, test = set(), set()
    for name, target in (("validation_list.txt", val), ("testing_list.txt", test)):
        f = sc_root / name
        if f.is_file():
            target.update(line.strip() for line in f.read_text().splitlines() if line.strip())
    return val, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/audio")
    ap.add_argument("--limit", type=int, default=0, help="process only N utterances (smoke)")
    ap.add_argument(
        "--source",
        choices=["tar", "hf"],
        default="tar",
        help="'tar': official tarball via urllib; 'hf': HuggingFace datasets "
        "(fallback for TLS-intercepting proxies).",
    )
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    from torchaudio.transforms import MelSpectrogram

    mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=MEL_BINS, hop_length=125)
    label_index = {c: i for i, c in enumerate(COMMANDS)}
    buckets = {"train": ([], []), "val": ([], [])}

    t0 = time.time()
    if args.source == "hf":
        import io

        import pyarrow.parquet as pq
        from huggingface_hub import list_repo_files, snapshot_download

        print("downloading google/speech_commands parquet (refs/convert/parquet) …", flush=True)
        local = snapshot_download(
            repo_id="google/speech_commands",
            repo_type="dataset",
            revision="refs/convert/parquet",
            allow_patterns="v0.02/*",
        )
        files = sorted(Path(local).rglob("*.parquet"))
        split_map = {"train": "train", "validation": "val", "test": "val"}
        n = 0
        for pf in files:
            hf_split = pf.parent.name
            split = split_map.get(hf_split)
            if split is None:
                continue
            table = pq.read_table(pf).to_pylist()
            for item in table:
                n += 1
                if args.limit and n > args.limit:
                    break
                label_name = item["file"].split("/")[0]  # the spoken word
                if label_name not in label_index:
                    continue
                with wave.open(io.BytesIO(item["audio"]["bytes"]), "rb") as w:
                    frames = w.readframes(w.getnframes())
                waveform = torch.frombuffer(frames, dtype=torch.int16).float() / 32768.0
                feats = mel(waveform).t()
                if feats.shape[0] < NUM_FRAMES:
                    feats = torch.nn.functional.pad(feats, (0, 0, 0, NUM_FRAMES - feats.shape[0]))
                feats = feats[:NUM_FRAMES]
                buckets[split][0].append(feats)
                buckets[split][1].append(label_index[label_name])
                if n % 5000 == 0:
                    print(f"  {n} files ({time.time() - t0:.0f}s)", flush=True)
    else:
        sc_root = _download_and_extract(root)
        val_list, test_list = _split_sets(sc_root)
        n = 0
        for label in COMMANDS:
            for wav_path in sorted((sc_root / label).glob("*.wav")):
                n += 1
                if args.limit and n > args.limit:
                    break
                waveform = _read_wav(wav_path)
                feats = mel(waveform).t()  # [time, mel]
                if feats.shape[0] < NUM_FRAMES:
                    feats = torch.nn.functional.pad(feats, (0, 0, 0, NUM_FRAMES - feats.shape[0]))
                feats = feats[:NUM_FRAMES]
                rel = f"{label}/{wav_path.name}"
                split = "val" if rel in val_list or rel in test_list else "train"
                buckets[split][0].append(feats)
                buckets[split][1].append(label_index[label])
                if n % 5000 == 0:
                    print(f"  {n} files ({time.time() - t0:.0f}s)", flush=True)

    for split, (xs, ys) in buckets.items():
        if not xs:
            print(f"WARNING: empty {split} split", flush=True)
            continue
        d = torch.utils.data.TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))
        out = root / f"{split}.pt"
        torch.save(d, out)
        print(f"{split}: {len(xs)} samples → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
