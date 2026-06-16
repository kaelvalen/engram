"""PTB-XL task → label mapping (pure logic, no pandas/wfdb/dataset needed)."""

from __future__ import annotations

import pytest
from prism.data.ptbxl_tasks import (
    TASKS,
    multi_hot,
    record_labels,
    task_vocab,
)

# Synthetic scp_statements.csv view: a few statements spanning diag/form/rhythm.
SCP = {
    "NORM": {
        "diagnostic": 1,
        "form": 0,
        "rhythm": 0,
        "diagnostic_subclass": "NORM",
        "diagnostic_class": "NORM",
    },
    "IMI": {
        "diagnostic": 1,
        "form": 0,
        "rhythm": 0,
        "diagnostic_subclass": "IMI",
        "diagnostic_class": "MI",
    },
    "AMI": {
        "diagnostic": 1,
        "form": 0,
        "rhythm": 0,
        "diagnostic_subclass": "AMI",
        "diagnostic_class": "MI",
    },
    "NDT": {
        "diagnostic": 1,
        "form": 0,
        "rhythm": 0,
        "diagnostic_subclass": "STTC",
        "diagnostic_class": "STTC",
    },
    "ABQRS": {
        "diagnostic": 0,
        "form": 1,
        "rhythm": 0,
        "diagnostic_subclass": float("nan"),
        "diagnostic_class": float("nan"),
    },
    "SR": {
        "diagnostic": 0,
        "form": 0,
        "rhythm": 1,
        "diagnostic_subclass": float("nan"),
        "diagnostic_class": float("nan"),
    },
    "AFIB": {
        "diagnostic": 0,
        "form": 0,
        "rhythm": 1,
        "diagnostic_subclass": float("nan"),
        "diagnostic_class": float("nan"),
    },
}


def test_vocab_sizes_per_task():
    assert task_vocab(SCP, "superdiag") == ["NORM", "MI", "STTC", "CD", "HYP"]
    assert task_vocab(SCP, "all") == sorted(SCP)
    assert task_vocab(SCP, "diag") == ["AMI", "IMI", "NDT", "NORM"]
    assert task_vocab(SCP, "form") == ["ABQRS"]
    assert task_vocab(SCP, "rhythm") == ["AFIB", "SR"]
    assert task_vocab(SCP, "subdiag") == ["AMI", "IMI", "NORM", "STTC"]


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown PTB-XL task"):
        task_vocab(SCP, "bogus")


def test_record_labels_superdiag_dedup_and_filter():
    # a record with two MI codes + a rhythm code → super-diag = {MI} only, deduped
    labels = record_labels(SCP, ["IMI", "AMI", "SR"], "superdiag")
    assert labels == ["MI"]


def test_record_labels_multilabel_superdiag():
    labels = record_labels(SCP, ["NORM", "NDT"], "superdiag")
    assert set(labels) == {"NORM", "STTC"}


def test_record_labels_form_and_rhythm():
    assert record_labels(SCP, ["ABQRS", "NORM"], "form") == ["ABQRS"]
    assert record_labels(SCP, ["SR", "AFIB", "NORM"], "rhythm") == ["SR", "AFIB"]


def test_record_labels_subdiag_uses_subclass():
    assert record_labels(SCP, ["IMI", "AMI"], "subdiag") == ["IMI", "AMI"]


def test_record_labels_all_keeps_known_codes():
    assert record_labels(SCP, ["NORM", "SR", "UNKNOWN"], "all") == ["NORM", "SR"]


def test_multi_hot():
    vocab = task_vocab(SCP, "superdiag")
    vec = multi_hot(["MI", "STTC"], vocab)
    assert vec == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert multi_hot([], vocab) == [0.0] * 5


@pytest.mark.parametrize("task", TASKS)
def test_every_task_roundtrips_to_finite_vector(task):
    vocab = task_vocab(SCP, task)
    labels = record_labels(SCP, list(SCP.keys()), task)
    vec = multi_hot(labels, vocab)
    assert len(vec) == len(vocab)
    assert all(v in (0.0, 1.0) for v in vec)
