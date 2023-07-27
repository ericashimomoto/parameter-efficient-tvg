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

class DORI(CHARADES_STA, YOUCOOKII, ANET_CAP):
    def __init__(self, args):
        dataset_name = args.pop("dataset_name")
        self.dataset_name = dataset_name
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

    def get_spatial_features_filepath(self, ann):
        if self.dataset_name == "CHARADES_STA":
            file_path = os.path.join(self.obj_feat_path, 
                        "{}.pkl".format(ann['video']))
            
        elif self.dataset_name == "ANET_CAP":
            file_path = os.path.join(self.obj_feat_path, 
                        "{}.pkl".format(ann['video']))
            
        elif self.dataset_name == "YOUCOOKII":
            file_path = os.path.join(self.obj_feat_path, 
                        ann['subset'], 
                        ann['recipe'], 
                        "{}.pkl".format(ann['video']))

        return file_path

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        ann = self.anns[index]

        selected_frames = self.selected_frames[ann['video']]

        ### Load object features
        object_features = []
        human_features  = []

        file_path = self.get_spatial_features_filepath(ann)

        with open(file_path, "rb") as f:
            features = pickle.load(f)
        
        for i, frame in enumerate(selected_frames):

            feature = features[frame]

            aux_obj = []
            aux_hum = []

            for indx, obj_type in enumerate(feature['object_class']):
                if self.mapping_obj[str(obj_type)]['human']:
                    aux_hum.append(feature['features'][indx])
                else:
                    aux_obj.append(feature['features'][indx])

            aux_obj = np.array(aux_obj)
            aux_hum = np.array(aux_hum)

            object_features.append(aux_obj)
            human_features.append(aux_hum)

        ### Load activity features
        i3dfeat = "{}/{}.npy".format(self.feature_path, ann['video'])
        i3dfeat = np.load(i3dfeat)
        i3dfeat = np.squeeze(i3dfeat)
        i3dfeat = torch.from_numpy(i3dfeat)
        feat_length = i3dfeat.shape[0]

        ### Get query representation
        if self.cfg.LANGUAGE.MODEL == "GLOVE":
            if self.is_training:
                raw_tokens = ann['tokens'][:self.train_max_length]
            else:
                raw_tokens = ann['tokens'][:self.test_max_length]

            indices = self.vocab.tokens2indices(raw_tokens)
            queries = [self.embedding_matrix[index] for index in indices]
            queries = torch.stack(queries)
        
        else:
            queries = ann['description']
        
        ### Get gold labels
        localization = np.zeros(feat_length, dtype=np.float32)
        start = math.floor(ann['feature_start'])
        end   = math.floor(ann['feature_end'])
        time_start = ann['time_start']
        time_end = ann['time_end']

        loc_start = np.ones(feat_length, dtype=np.float32) * self.epsilon
        loc_end   = np.ones(feat_length, dtype=np.float32) * self.epsilon
        y = (1 - (feat_length-3) * self.epsilon - 0.5)/ 2

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

        return index, i3dfeat, object_features, human_features, queries, torch.from_numpy(loc_start), torch.from_numpy(loc_end), \
               torch.from_numpy(localization), time_start, time_end, ann['number_frames']/ann['number_features'], ann['fps']
