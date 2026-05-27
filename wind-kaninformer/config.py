import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEASONS_DIR = os.path.join(BASE_DIR, 'seasons')
DECOMP_DIR = os.path.join(BASE_DIR, 'decomposition')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

RAW_CSV = os.path.join(DATA_DIR, 'hourly.csv')

SEASON_DATES = {
    'winter': ('2020-12-01', '2021-02-28'),
    'spring': ('2021-03-01', '2021-05-31'),
    'summer': ('2021-06-01', '2021-08-31'),
    'autumn': ('2021-09-01', '2021-11-30'),
}

SEASON_ORDER = ['spring', 'summer', 'autumn', 'winter']

ALL_METEO_VARS = ['ET', 'PCP', 'SR', 'VP', 'AT', 'RH', 'DPT', 'WD', 'ST']
TARGET_VAR = 'WS'

PCC_THRESHOLD = 0.5
PCC_PVALUE = 0.05
SE_THRESHOLD = 0.4
N_EWT = 8

VMD_ALPHA = 2000
VMD_TAU = 0
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7
VMD_K_RANGE = range(2, 16)
VMD_RER_THRESHOLD = 0.03

EXPECTED_K = {'spring': 11, 'summer': 10, 'autumn': 11, 'winter': 10}

EXPECTED_FEATURES = {
    'spring': ['ET'],
    'summer': ['ET', 'AT', 'RH'],
    'autumn': ['ET'],
    'winter': [],
}

N_STEP = 7
N_OUT = 3
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

D_MODEL = 64
N_HEADS = 8
E_LAYERS = 2
D_LAYERS = 1
FACTOR = 5
DROPOUT = 0.05
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 200
PATIENCE = 3
KAN_GRID = 5
KAN_K = 3

RANDOM_SEED = 42

# Will be populated by step2_pcc_selection.py at runtime
SEASON_FEATURES = {}

EXPECTED_RESULTS = {
    'spring': {
        1: {'rmse': 0.641, 'mae': 0.509, 'mape': 18.2},
        2: {'rmse': 0.821, 'mae': 0.645, 'mape': 23.1},
        3: {'rmse': 0.910, 'mae': 0.729, 'mape': 26.6},
    },
    'summer': {
        1: {'rmse': 0.424, 'mae': 0.325, 'mape': 18.3},
        2: {'rmse': 0.505, 'mae': 0.392, 'mape': 21.9},
        3: {'rmse': 0.518, 'mae': 0.407, 'mape': 23.2},
    },
    'autumn': {
        1: {'rmse': 0.495, 'mae': 0.330, 'mape': 22.9},
        2: {'rmse': 0.764, 'mae': 0.467, 'mape': 27.9},
        3: {'rmse': 0.978, 'mae': 0.562, 'mape': 31.2},
    },
    'winter': {
        1: {'rmse': 0.725, 'mae': 0.527, 'mape': 25.5},
        2: {'rmse': 0.969, 'mae': 0.693, 'mape': 30.6},
        3: {'rmse': 1.093, 'mae': 0.784, 'mape': 34.4},
    },
}
