import os
import json
import time
import math
import torch
import pickle
import numpy as np
from random import shuffle
import random

from .anet_cap import ANET_CAP
from .youcookII import YOUCOOKII
from .charades_sta import CHARADES_STA

class TMLGA(CHARADES_STA, YOUCOOKII, ANET_CAP):
    def __init__(self, args):
        dataset_name = args.pop("dataset_name")
        if dataset_name == "CHARADES_STA":
            CHARADES_STA.__init__(self, **args)
            CHARADES_STA.createIndex(self)
        elif dataset_name == "YOUCOOKII":
            YOUCOOKII.__init__(self, **args)
            YOUCOOKII.createIndex(self)
        elif dataset_name == "ANET_CAP":
            ANET_CAP.__init__(self, **args)
            ANET_CAP.createIndex(self)
        self.epsilon = 1e-5
        self.ids = list(self.anns.keys())

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):    
        ann = self.anns[index]
        
        ### Load activity features
        i3dfeat = "{}/{}.npy".format(self.feature_path, ann['video'])
        i3dfeat = np.load(i3dfeat)
        i3dfeat = np.squeeze(i3dfeat)
        i3dfeat = torch.from_numpy(i3dfeat)
        feat_length = i3dfeat.shape[0]

        ### Get query representation
        if self.cfg.LANGUAGE.MODEL == "glove":
            if self.is_training:
                raw_tokens = ann['tokens'][:self.train_max_length]
            else:
                raw_tokens = ann['tokens'][:self.test_max_length]

            indices = self.vocab.tokens2indices(raw_tokens)
            queries = [self.embedding_matrix[index] for index in indices]
            queries = torch.stack(queries)
        
        # elif self.cfg.LANGUAGE.MODEL == "llama":
        #     query_emb = "{}/{}.pth".format(self.llama_emb_path, ann['video'])
        #     query_emb = torch.load(query_emb)
        #     queries = query_emb
        else:
            queries = ann['description']

        if ann['augmentation'] == 1:
            feature_start = ann['feature_start']
            feature_end   = ann['feature_end']

            offset = int(math.floor(feature_start))
            if offset != 0:
                offset = np.random.randint(0, int(round(feature_start)))

            new_feature_start = feature_start - offset
            new_feature_end   = feature_end - offset

            i3dfeat = i3dfeat[offset:,:]

            feat_length = ann['number_features'] - offset
            localization = np.zeros(feat_length, dtype=np.float32)

            start = math.floor(new_feature_start)
            end   = math.floor(new_feature_end)

            time_start = (new_feature_start * ann['number_frames']/ann['number_features']) / ann['fps']
            time_end = (new_feature_end * ann['number_frames']/ann['number_features']) / ann['fps']
            time_offset = (offset * ann['number_frames']/ann['number_features']) / ann['fps']


        else:
            localization = np.zeros(feat_length, dtype=np.float32)

            # loc_start =
            start = math.floor(ann['feature_start'])
            end   = math.floor(ann['feature_end'])
            time_start = ann['time_start']
            time_end = ann['time_end']

        #if end - start == 0:
            #print(ann)

        # print(start, end, feat_length, ann['augmentation'])

        loc_start = np.ones(feat_length, dtype=np.float32) * self.epsilon
        loc_end   = np.ones(feat_length, dtype=np.float32) * self.epsilon
        y = (1 - (feat_length-3) * self.epsilon - 0.5)/ 2
        # print(y)
        if start > 0:
            loc_start[start - 1] = y
        if start < feat_length-1:
            loc_start[start + 1] = y
        loc_start[start] = 0.5

        if end > 0:
            loc_end[end - 1] = y
        if end < feat_length-1:
            loc_end[end + 1] = y
        loc_end[end] = 0.5

        y = 1.0
        localization[start:end] = y

        # return index, i3dfeat, tokens, torch.from_numpy(loc_start), torch.from_numpy(loc_end), torch.from_numpy(localization),\
        #        time_start, time_end, ann['number_frames']/ann['number_features'], ann['fps'] FOR KL

        return index, i3dfeat, queries, torch.from_numpy(loc_start), torch.from_numpy(loc_end), torch.from_numpy(localization),\
                   time_start, time_end, ann['number_frames']/ann['number_features'], ann['fps']