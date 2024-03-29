import os
from yacs.config import CfgNode as CN

_C = CN()

##############
#### MISC ####
_C.EXPERIMENT_NAME = ""
_C.ENGINE_STAGE = "TRAINER"
_C.LOG_DIRECTORY = ""
_C.VISUALIZATION_DIRECTORY = ""
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

_C.EARLY_STOPPER = CN()
_C.EARLY_STOPPER.PATIENCE = 3
_C.EARLY_STOPPER.MIN_DELTA = 0

##############
#### TEST ####
_C.TEST = CN()
_C.TEST.CHECKPOINT = ""

#### TRAIN ####
_C.TRAIN = CN()
_C.TRAIN.CHECKPOINT = ""

##################
#### Dataset #####

_C.DATASETS = CN()
_C.DATASETS.NUMBER_CLASSES = 200
_C.DATASETS.PORTION_OF_DATA = 1.

_C.DATASETS.TRAIN = ""
_C.DATASETS.TEST = ""
_C.DATASETS.TRAIN_SAMPLES = 0.
_C.DATASETS.TEST_SAMPLES = 0.

##################
#### Sentence #####

_C.SENTENCE = CN()
_C.SENTENCE.MIN_COUNT = 5
_C.SENTENCE.TRAIN_MAX_LENGTH = 30
_C.SENTENCE.TEST_MAX_LENGTH = 30

##################
#### Language Module #####
_C.LANGUAGE = CN()
_C.LANGUAGE.MODEL = "glove"
_C.LANGUAGE.LOAD_FROM = ""
_C.LANGUAGE.TRAINING = "no-finetuning" # "finetuning", "adapter"
_C.LANGUAGE.ADAPTER_TYPE = "Pfeiffer"

##### Models ######
_C.MODEL = "EXCL"

###### ExCL #######
_C.VIDEO_LSTM = CN(new_allowed=True)
_C.LANGUAGE_LSTM = CN(new_allowed=True)

###################
###### TMLGA #######
_C.DYNAMIC_FILTER = CN(new_allowed=True)
_C.DYNAMIC_FILTER.TAIL_MODEL = "LSTM"
_C.DYNAMIC_FILTER.POOLING    = "MeanPoolingLayer"
_C.DYNAMIC_FILTER.HEAD_MODEL = "MLP"

###################
###### DORi #######
_C.SPATIAL_GRAPH = CN(new_allowed=True)
_C.SPATIAL_GRAPH.NUMBER_ITERATIONS = 2
_C.SPATIAL_GRAPH.OUTPUT_SIZE = 512
_C.SPATIAL_GRAPH.TAIL_MODEL = "GRU"
_C.SPATIAL_GRAPH.POOLING    = "MeanPoolingLayer"

############################
###### TMLGA and DORi #######
_C.LOSS = CN()
_C.LOSS.ATTENTION = True

############################
###### All Model #######
_C.REDUCTION = CN()
_C.REDUCTION.INPUT_SIZE = 1024
_C.REDUCTION.OUTPUT_SIZE = 512

_C.LOCALIZATION = CN()
_C.LOCALIZATION.INPUT_SIZE = 512
_C.LOCALIZATION.HIDDEN_SIZE = 256
_C.LOCALIZATION.NUM_LAYERS = 2
_C.LOCALIZATION.BIAS = False
_C.LOCALIZATION.DROPOUT = 0.5
_C.LOCALIZATION.BIDIRECTIONAL = True
_C.LOCALIZATION.BATCH_FIRST = True

_C.CLASSIFICATION = CN()
_C.CLASSIFICATION.INPUT_SIZE = 512
_C.CLASSIFICATION.OUTPUT_SIZE = 1


###################
#### OPTIMIZER ####

_C.SOLVER = CN(new_allowed=True)
_C.SOLVER.TYPE = "ADAM"
_C.SOLVER.EPSILON = 0.1
_C.SOLVER.SCHEDULER = "EPOCH_DECAY"
_C.SOLVER.SCH_STEP_SIZE = 6
_C.SOLVER.SCH_GAMMA = 0.1


####################
#### EXPERIMENT ####
_C.BATCH_SIZE_TRAIN = 32
_C.BATCH_SIZE_TEST = 32
_C.NUM_WORKERS_TRAIN = 5
_C.NUM_WORKERS_TEST = 5
_C.EPOCHS = 10