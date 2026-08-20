from __future__ import annotations

from engram.integrations.huggingface import get_engram_hf_classes, is_transformers_available


def test_hf_class_factory_matches_transformers_install():
    pair = get_engram_hf_classes()
    if is_transformers_available():
        assert pair is not None
        cfg_cls, model_cls = pair
        assert cfg_cls.__name__ == "ENGRAMConfigHF"
        assert model_cls.__name__ == "ENGRAMPreTrainedForClassification"
    else:
        assert pair is None
