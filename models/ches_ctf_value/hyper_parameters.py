from . import state_space_parameters as ssp
import metaqnn.data_loader as data_loader
import numpy as np

MODEL_NAME = 'CHES_CTF_Value'

# Number of output neurons
NUM_CLASSES = 256  # Number of output neurons

# Input Size
INPUT_SIZE = 2200

# Batch Queue parameters
TRAIN_BATCH_SIZE = 128  # Batch size for training (scaled linearly with number of gpus used)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45000  # Number of training examples
VALIDATION_FROM_ATTACK_SET = True
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2500  # Number of validation examples


MAX_EPOCHS = 50  # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 5e-3  # The max LR (scaled linearly with number of gpus used)

# Bulk data folder
BULK_ROOT = 'data/CHES_CTF/experiment_value/'
DATA_ROOT = BULK_ROOT + '../data/'

# Trained model dir
TRAINED_MODEL_DIR = BULK_ROOT + 'trained_models'
DB_FILE = DATA_ROOT + 'ches_ctf.h5'

(TRAIN_TRACES, TRAIN_DATA), (ATTACK_TRACES, ATTACK_DATA) = data_loader.load_ches_hd5(
    DB_FILE,
    '/Profiling_traces/traces', '/Profiling_traces/metadata',
    '/Attack_traces/traces', '/Attack_traces/metadata'
)

TRAIN_LABELS = np.load(DATA_ROOT + 'train_labels.npy')
ATTACK_LABELS = np.load(DATA_ROOT + 'attack_labels.npy')

KEY = np.load(DATA_ROOT + 'attack_key.npy')
ATTACK_KEY_BYTE = 0
ATTACK_PRECOMPUTED_BYTE_VALUES = np.load(DATA_ROOT + f'attack_precomputed_byte{ATTACK_KEY_BYTE}_values.npy')

TRACES_PER_ATTACK = 2000  # Maximum number of traces to use per attack
NUM_ATTACKS = 100  # Number of attacks to average the GE over
