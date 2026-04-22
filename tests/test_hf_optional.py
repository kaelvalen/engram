from __future__ import annotations

from prism.integrations.huggingface import get_prism_hf_classes, is_transformers_available


def test_hf_class_factory_matches_transformers_install():
    pair = get_prism_hf_classes()
    if is_transformers_available():
        assert pair is not None
        cfg_cls, model_cls = pair
        assert cfg_cls.__name__ == "PRISMConfigHF"
        assert model_cls.__name__ == "PRISMPreTrainedForClassification"
    else:
        assert pair is None
