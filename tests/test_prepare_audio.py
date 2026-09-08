from scripts.prepare_audio import _split_for_path


def test_audio_test_split_is_not_folded_into_validation():
    val = {"yes/val.wav"}
    test = {"yes/test.wav"}
    assert _split_for_path("yes/val.wav", val, test) == "val"
    assert _split_for_path("yes/test.wav", val, test) == "test"
    assert _split_for_path("yes/train.wav", val, test) == "train"


def test_audio_test_split_wins_if_lists_overlap():
    assert _split_for_path("yes/shared.wav", {"yes/shared.wav"}, {"yes/shared.wav"}) == "test"
