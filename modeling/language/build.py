from transformers import BertModel, RobertaModel, DebertaModel, DebertaV2Model
from transformers.adapters import HoulsbyConfig, PfeifferConfig , ParallelConfig, PfeifferInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig

from config.settings import RESULTS_PATH


def get_language_model(cfg):

    if cfg.LANGUAGE.MODEL == "bert":
        print("Using BERT")
        language_model = BertModel.from_pretrained(
            cfg.LANGUAGE.LOAD_FROM, cache_dir=RESULTS_PATH,
        )
    elif cfg.LANGUAGE.MODEL == "roberta":
        print("Using RoBERTa")
        language_model = RobertaModel.from_pretrained(
            cfg.LANGUAGE.LOAD_FROM, cache_dir=RESULTS_PATH,
        )
    elif cfg.LANGUAGE.MODEL == "deberta":
        print("Using DeBERTa")
        if "V3" in cfg.LANGUAGE.LOAD_FROM or "V2" in cfg.LANGUAGE.LOAD_FROM:
            language_model = DebertaV2Model.from_pretrained(
                cfg.LANGUAGE.LOAD_FROM, cache_dir=RESULTS_PATH,
            )
        else:
            language_model = DebertaModel.from_pretrained(
                cfg.LANGUAGE.LOAD_FROM, cache_dir=RESULTS_PATH)

    if cfg.LANGUAGE.TRAINING == "no-finetuning":
        # Freeze LM parameters
        for p in language_model.parameters(): p.requires_grad=False
    elif cfg.LANGUAGE.TRAINING == "adapter":
        print("Using adapter: {}".format(cfg.LANGUAGE.ADAPTER_TYPE))
        # Train only the adapter weights
        if cfg.LANGUAGE.ADAPTER_TYPE == "Pfeiffer":
            config = PfeifferConfig ()
        elif cfg.LANGUAGE.ADAPTER_TYPE == "Houlsby":
            config = HoulsbyConfig()
        elif cfg.LANGUAGE.ADAPTER_TYPE== "Parallel":
            config = ParallelConfig()
        elif cfg.LANGUAGE.ADAPTER_TYPE== "Inverse":
            config = PfeifferInvConfig()
        elif cfg.LANGUAGE.ADAPTER_TYPE== "Prefix":
            config = PrefixTuningConfig()
        elif cfg.LANGUAGE.ADAPTER_TYPE== "Compacter":
            config = CompacterConfig()
        elif cfg.LANGUAGE.ADAPTER_TYPE== "LoRA":
            config = LoRAConfig()

        language_model.add_adapter("tvg_adapter", config=config)
        language_model.train_adapter("tvg_adapter")
    
    return language_model