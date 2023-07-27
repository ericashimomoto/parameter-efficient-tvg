import torch
import numpy as np
from utils import rnns

class DORI(object):

    def __init__(self, cfg, tokenizer, max_len):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        
        transposed_batch = list(zip(*batch))

        index      = transposed_batch[0]
        videoFeat  = transposed_batch[1]
        objectFeat = transposed_batch[2]
        humanFeat  = transposed_batch[3]
        queries    = transposed_batch[4] # will be tokens, if Glove; sentence, if PLM
        start      = transposed_batch[5]
        end        = transposed_batch[6]
        localiz    = transposed_batch[7]
        time_start = transposed_batch[8]
        time_end   = transposed_batch[9]
        factor     = transposed_batch[10]
        fps        = transposed_batch[11]

        objectFeat, objectFeat_lengths = rnns.pad_spatial_sequence(objectFeat)
        humanFeat, humanFeat_lengths = rnns.pad_spatial_sequence(humanFeat)            
        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat)

        video_info = {}
        video_info["objectFeat"]         = objectFeat
        video_info["objectFeat_lengths"] = objectFeat_lengths
        video_info["humanFeat"]          = humanFeat
        video_info["humanFeat_lengths"]  = humanFeat_lengths
        video_info["videoFeat"] = videoFeat
        video_info["videoFeat_lengths"]  = videoFeat_lengths


        localiz, localiz_lengths = rnns.pad_sequence(localiz)
        start, start_lengths = rnns.pad_sequence(start)
        end, end_lengths     = rnns.pad_sequence(end)

        if self.tokenizer:
            tokenizer_output = self.tokenizer(
                queries,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                return_length=True,
            )

            tokens_ids = tokenizer_output["input_ids"]
            tokens_lengths = tokenizer_output["length"]
            tokens_type_ids = tokenizer_output.get("token_type_ids", None)
            position_ids = (
                torch.arange(1, tokens_ids.shape[1] + 1)
                .unsqueeze(0)
                .repeat(tokens_ids.shape[0], 1)
            )
            lang_mask = tokenizer_output["attention_mask"]

            query_info = {}
            query_info["tokens_ids"]      = tokens_ids
            query_info["tokens_lengths"]  = tokens_lengths
            query_info["tokens_type_ids"] = tokens_type_ids
            query_info["position_ids"]    = position_ids
            query_info["lang_mask"]       = lang_mask
        
        else:
            tokens, tokens_lengths   = rnns.pad_sequence(queries)

            query_info = {}
            query_info["tokens"]         = tokens
            query_info["tokens_lengths"] = tokens_lengths

        return index, \
               video_info, \
               query_info, \
               start,  \
               end, \
               localiz, \
               localiz_lengths, \
               time_start, \
               time_end, \
               factor, \
               fps, \

            