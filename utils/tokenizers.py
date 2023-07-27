from transformers import BertTokenizer, DebertaTokenizer, RobertaTokenizer, DebertaV2Tokenizer

def get_tokenizer(cfg):

    if cfg.LANGUAGE.MODEL == "bert":
        tokenizer = BertTokenizer.from_pretrained(cfg.LANGUAGE.LOAD_FROM)
    elif cfg.LANGUAGE.MODEL == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.LANGUAGE.LOAD_FROM)
    elif cfg.LANGUAGE.MODEL == "deberta":
        if "V3" in cfg.LANGUAGE.LOAD_FROM or "V2" in cfg.LANGUAGE.LOAD_FROM:
            tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.LANGUAGE.LOAD_FROM)
        else:
            # Hardcoded to use "microsoft/deberta-base"
            # tokenizer = DebertaTokenizer.from_pretrained(cfg.LANGUAGE.LOAD_FROM)
            tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    else:
        raise RuntimeError("Language Model not available: {}".format(cfg.LANGUAGE.MODEL))
    
    return tokenizer