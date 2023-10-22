# General Constants:
LABEL_NAME = 'label'
TEXT_COLUMN_NAME = [f"sentence_{i}" for i in range(1, 10)]

# SSPabs Constants:
SSPABS = 'SSPabs'
SSPABS_TRAIN_NAME = 'train.txt'
SSPABS_VALID_NAME = 'valid.txt'
SSPABS_TEST_NAME = 'test.txt'
SSPABS_DATA_DIR = 'data/SSP/abs'
SSPABS_LABELS = ["0", "1"]
SSPABS_TEXT_COLUMNS = 1

# PDTB Constants:
PDTB_I = 'PDTB-I'
PDTB_E = 'PDTB-E'
PDTB_TRAIN_NAME = 'train.txt'
PDTB_VALID_NAME = 'valid.txt'
PDTB_TEST_NAME = 'test.txt'
PDTB_DATA_DIR = 'data/PDTB'
PDTB_DIRS = {PDTB_E: 'Explicit', PDTB_I: 'Implicit'}
PDTB_E_LABELS = [
    'Comparison.Concession',
    'Comparison.Contrast',
    'Contingency.Cause',
    'Contingency.Condition',
    'Contingency.Pragmatic condition',
    'Expansion.Alternative',
    'Expansion.Conjunction',
    'Expansion.Instantiation',
    'Expansion.List',
    'Expansion.Restatement',
    'Temporal.Asynchronous',
    'Temporal.Synchrony',
]
PDTB_I_LABELS = [
    'Comparison.Concession',
    'Comparison.Contrast',
    'Contingency.Cause',
    'Contingency.Pragmatic cause',
    'Expansion.Alternative',
    'Expansion.Conjunction',
    'Expansion.Instantiation',
    'Expansion.List',
    'Expansion.Restatement',
    'Temporal.Asynchronous',
    'Temporal.Synchrony',
]
PDTB_E_TEXT_COLUMNS = 2
PDTB_I_TEXT_COLUMNS = 2


# SP Constants:
SPARXIV = 'SParxiv'
SPROCSTORY = 'SProcstory'
SPWIKI = 'SPwiki'
SP_TRAIN_NAME = 'train.txt'
SP_VALID_NAME = 'valid.txt'
SP_TEST_NAME = 'test.txt'
SP_DATA_DIR = 'data/SP'
SP_DIRS = {SPARXIV: 'arxiv', SPROCSTORY: 'rocstory', SPWIKI: 'wiki'}
SP_LABELS = ["0", "1", "2", "3", "4"]
SP_TEXT_COLUMNS = 5

# BSO Constants:
BSOARXIV = 'BSOarxiv'
BSOROCSTORY = 'BSOrocstory'
BSOWIKI = 'BSOwiki'
BSO_TRAIN_NAME = 'train.txt'
BSO_VALID_NAME = 'valid.txt'
BSO_TEST_NAME = 'test.txt'
BSO_DATA_DIR = 'data/BSO'
BSO_DIRS = {BSOARXIV: 'arxiv', BSOROCSTORY: 'rocstory', BSOWIKI: 'wiki'}
BSO_LABELS = ["0", "1"]
BSO_TEXT_COLUMNS = 2

# DC Constants:
DCCHAT = 'DCchat'
DCWIKI = 'DCwiki'
DC_TRAIN_NAME = 'train.txt'
DC_VALID_NAME = 'valid.txt'
DC_TEST_NAME = 'test.txt'
DC_DATA_DIR = 'data/DC'
DC_DIRS = {DCCHAT: 'chat', DCWIKI: 'wiki'}
DC_LABELS = ["0", "1"]
DC_TEXT_COLUMNS = 6


# RST Constants:
RST = 'RST'
RST_TRAIN_NAME = 'RST_TRAIN.pkl'
RST_VALID_NAME = 'RST_DEV.pkl'
RST_TEST_NAME = 'RST_TEST.pkl'
RST_DATA_DIR = 'data/RST'
RST_LABELS = [
    'NS-Explanation',
    'NS-Evaluation',
    'NN-Condition',
    'NS-Summary',
    'SN-Cause',
    'SN-Background',
    'NS-Background',
    'SN-Summary',
    'NS-Topic-Change',
    'NN-Explanation',
    'SN-Topic-Comment',
    'NS-Elaboration',
    'SN-Attribution',
    'SN-Manner-Means',
    'NN-Evaluation',
    'NS-Comparison',
    'NS-Contrast',
    'SN-Condition',
    'NS-Temporal',
    'NS-Enablement',
    'SN-Evaluation',
    'NN-Topic-Comment',
    'NN-Temporal',
    'NN-Textual-organization',
    'NN-Same-unit',
    'NN-Comparison',
    'NN-Topic-Change',
    'SN-Temporal',
    'NN-Joint',
    'SN-Enablement',
    'SN-Explanation',
    'NN-Contrast',
    'NN-Cause',
    'SN-Contrast',
    'NS-Attribution',
    'NS-Topic-Comment',
    'SN-Elaboration',
    'SN-Comparison',
    'NS-Cause',
    'NS-Condition',
    'NS-Manner-Means'
]
RST_TEXT_COLUMNS = 2
