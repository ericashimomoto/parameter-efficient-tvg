import os
import json
import time
import math
import torch
import pickle
import numpy as np
from random import shuffle

from utils.vocab import Vocab
from utils.sentence import get_embedding_matrix
from utils.tokenizers import get_tokenizer

from torch.utils.data import Dataset

from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


class ANET_CAP(Dataset):

    def __init__(self, features_path,
                       ann_file_path,
                       obj_feat_path,
                       embeddings_path,
                       cfg):

        self.feature_path = features_path
        self.ann_file_path = ann_file_path
        self.obj_feat_path = obj_feat_path
        self.cfg = cfg

        self.is_training = 'training' in ann_file_path
        print(self.is_training)

        print('loading annotations into memory...', end=" ")
        tic = time.time()
        aux = json.load(open(ann_file_path, 'r'))
        self.dataset = aux['annotations']

        # self.glove = np.load(vocab_glove, allow_pickle=True).item()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        self.min_count = self.cfg.SENTENCE.MIN_COUNT
        self.train_max_length = self.cfg.SENTENCE.TRAIN_MAX_LENGTH
        self.test_max_length = self.cfg.SENTENCE.TEST_MAX_LENGTH

        if cfg.LANGUAGE.MODEL == "glove":
        
            vocab_file_name = f'anet_vocab_{self.min_count}_{self.train_max_length}.pickle'

            self.vocab_file_path = vocab_file_name
            print(self.vocab_file_path)
            self.create_vocab()

            embeddings_file_name = f'anet_embeddings_{self.min_count}_{self.train_max_length}.pth'
            self.embeddings_file_path = embeddings_file_name
            self.get_embedding_matrix(embeddings_path)
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(self.cfg)

        # self.createIndex()
        # self.ids   = list(self.anns.keys())
        # self.epsilon = 1E-10

    def tIoU(self, start, end, pred_start, pred_end):
            tt1 = np.maximum(start, pred_start)
            tt2 = np.minimum(end, pred_end)
            # Intersection including Non-negative overlap score.
            segments_intersection = (tt2 - tt1).clip(0)
            # Segment union.
            segments_union = (pred_end - pred_start) \
            + (end - start) - segments_intersection
            # Compute overlap as the ratio of the intersection
            # over union of two segments.
            tIoU = segments_intersection.astype(float) / segments_union
            return tIoU

    def create_vocab(self):

        if self.is_training:
            if not os.path.exists(self.vocab_file_path):
                print("Creating vocab")
                self.vocab = Vocab(
                    add_bos=False,
                    add_eos=False,
                    add_padding=False,
                    min_count=self.min_count)

                for example in self.dataset:
                    self.vocab.add_tokenized_sentence(example['tokens'][:self.train_max_length])

                self.vocab.finish()

                with open(self.vocab_file_path, 'wb') as f:
                    pickle.dump(self.vocab, f)
            else:
                with open(self.vocab_file_path, 'rb') as f:
                    self.vocab = pickle.load(f)

        else:
            print("Cargando vocab")
            with open(self.vocab_file_path, 'rb') as f:
                self.vocab = pickle.load(f)


    def get_embedding_matrix(self, embeddings_path):
        '''
        Gets you a torch tensor with the embeddings
        in the indices given by self.vocab.

        Unknown (unseen) words are each mapped to a random,
        different vector.


        :param embeddings_path:
        :return:
        '''
        if self.is_training and not os.path.exists(self.embeddings_file_path):
            tic = time.time()

            print('loading embeddings into memory...', end=" ")

            if 'glove' in embeddings_path.lower():
                tmp_file = get_tmpfile("test_word2vec.txt")
                _ = glove2word2vec(embeddings_path, tmp_file)
                embeddings = KeyedVectors.load_word2vec_format(tmp_file)
            else:
                embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

            print('Done (t={:0.2f}s)'.format(time.time() - tic))

            embedding_matrix = get_embedding_matrix(embeddings, self.vocab)

            with open(self.embeddings_file_path, 'wb') as f:
                torch.save(embedding_matrix, f)

        else:
            with open(self.embeddings_file_path, 'rb') as f:
                embedding_matrix  = torch.load(f)

        self.embedding_matrix = embedding_matrix


    def createIndex(self):
        print("Creating index..", end=" ")
        anns = {}
        size = int(round(len(self.dataset) * 1.))
        counter = 0
        for row in self.dataset[:size]:

            oIoU = self.tIoU(float(row['feature_start']), float(row['feature_end']), 0, float(row['number_features']))
            if self.is_training:

                #if oIoU > 0.9:
                    #continue
                if float(row['number_features']) < 10:
                    continue            # print(row) 
                if float(row['number_features']) >= 1200:
                    continue            # print(row)
            if float(row['feature_start']) > float(row['feature_end']):
                # print(row)
                continue
            #if math.floor(float(row['feature_start'])) - math.floor(float(row['feature_end'])) == 0:
                # print(row)
                #continue
            if math.floor(float(row['feature_end'])) >= float(row['number_features']):
                row['feature_end'] = float(row['number_features'])-1

            if self.is_training:
                #if oIoU < 0.9:
                    #if row['feature_start'] > 10:
                row['augmentation'] = 1
                anns[counter] = row.copy()
                counter += 1

            row['augmentation'] = 0
            anns[counter] = row
            counter+=1
        self.anns = anns
        print(" Ok! {}".format(len(anns.keys())))

