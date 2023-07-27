import logging

from . import batchcollators as BatchCollators
from . import datasets as Datasets
from config.paths_catalog import DatasetCatalog

from torch.utils.data import DataLoader
from .collate_batch import BatchCollator
from utils.imports import import_file

def build_dataset(model, dataset_name, cfg):
    datasets = []
    data = DatasetCatalog.get(dataset_name)
    factory = getattr(Datasets, model)
    arguments = data["args"]
    arguments["dataset_name"] = data["factory"]
    arguments["cfg"] = cfg

    dataset = factory(arguments)
    return dataset

def build_collator(method, args):
    factory = getattr(BatchCollators, method)
    return factory(**args)

def make_dataloader(cfg, is_train):

    # paths_catalog = import_file("config.paths_catalog", cfg.PATHS_CATALOG, True)
    # DatasetCatalog = paths_catalog.DatasetCatalog
    if is_train == True:
        dataset_name = cfg.DATASETS.TRAIN
        dataset = build_dataset(cfg.MODEL, dataset_name, cfg)
        # dataset = build_dataset(dataset_name,
        #                         DatasetCatalog,
        #                         cfg)
        collator_args = {"cfg": cfg,
                        "tokenizer": dataset.tokenizer,
                        "max_len": cfg.SENTENCE.TRAIN_MAX_LENGTH}
                        #"is_train": True}
        collator = build_collator(method=cfg.MODEL, args=collator_args)

        dataloader = DataLoader(dataset,
                                batch_size=cfg.BATCH_SIZE_TRAIN,
                                shuffle=is_train,
                                num_workers=cfg.NUM_WORKERS_TRAIN,
                                collate_fn=collator)
    else:
        dataset_name = cfg.DATASETS.TEST
        dataset = build_dataset(cfg.MODEL, dataset_name, cfg)
        collator_args = {"cfg": cfg,
                            "tokenizer": dataset.tokenizer,
                            "max_len": cfg.SENTENCE.TEST_MAX_LENGTH}
                            #"is_train": False}
        collator = build_collator(cfg.MODEL, collator_args)
        
        dataloader = DataLoader(dataset,
                                batch_size=cfg.BATCH_SIZE_TEST,
                                shuffle=is_train,
                                num_workers=cfg.NUM_WORKERS_TEST,
                                collate_fn=collator)

    return dataloader, len(dataset)
