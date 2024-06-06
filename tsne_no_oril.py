
import sys 
sys.path.insert(0, '../../')

import time
import os,io


import numpy as np
import pandas as pd
import random

from transformers import BertConfig
from common.common import create_folder
from common.pytorch import load_model
from dataLoader.utils import seq_padding,position_idx,index_seg,random_mask,gen_vocab
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from preprocess.standard_config import * 

from torch import nn
from dataLoader.Loader_for_test_downstream_classifier import MLMLoader
from model.mlm import EdwardsForMultiLabelPrediction

from model.optimiser import adam
import sklearn.metrics as skm

class EdwardsConfig(BertConfig):
    def __init__(self, config):
        super(EdwardsConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            seg_vocab_size = config.get('seg_vocab_size'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.vocab_size=config.get('vocab_size')
        self.demo_vocab_size = config.get('demo_vocab_size')
        
class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')

file_config = {
    'core_vocab':'',  # vocabulary idx2token, token2idx
    'data': './data/data_w_date_new_v5.csv',  # formated data 
    'model_path': './sessions/folli_downstream', # where to save model
    'model_name': 'edwards', # model name
    'file_name': 'edwards_tsne_no_oril.log',  # log path
}
create_folder(file_config['model_path'])


global_params = {
    'max_seq_len': 128,
    'min_visit': 2,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
}

optim_param = {
    'lr': 4.35821363648534e-05,
    'warmup_proportion': 0.1,
    'weight_decay': 0.00025139282767006803
}

train_params = {
    'batch_size': 11,
    'use_cuda': True,
    'max_len_seq': global_params['max_seq_len'],
    'device': 'cuda:0',
    'type': 'folli',
}


BertVocab = gen_vocab(CORE_EMBEDDING_CODE_VOC)
BertDecoder = dict((v,k) for k,v in BertVocab.items() )
DemoVocab = gen_vocab(DEMOGRAFIC_EMBEDDING_CODE_VOC)

model_config = {
    'vocab_size': len(BertVocab), # number of disease + symbols for word embedding
    'hidden_size': 228, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'demo_vocab_size': len(DemoVocab), # number of vocab for age embedding
    'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 5, # number of multi-head attention layers required
    'num_attention_heads': 6, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 1602, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
    'is_decoder': False,
}

conf = EdwardsConfig(model_config)

edwards_model = EdwardsForMultiLabelPrediction(conf, len(CODE_SUB_GROUP_W_KEYS[train_params['type']]))
output_model_file = os.path.join(file_config['model_path'], file_config['model_name'])
edwards_model = load_model(output_model_file, edwards_model)
edwards_model = edwards_model.to(global_params['device'])
edwards_model.eval()

label = [
'[CLS]',
'[SEP]',
'fsh: 0-5',
'fsh: 5-15',
'fsh: 15-30',
'fsh: 30-40',
'fsh: 40-',
'e2: 0-50',
'e2: 50-100',
'e2: 100-200',
'e2: 200-500',
'e2: 500-1000',
'e2: 1000-1500',
'e2: 1500-2000',
'e2: 2000-3000',
'e2: 3000-',
'lh: 0-1',
'lh: 1-1.5',
'lh: 1.5-2',
'lh: 2-3',
'lh: 3-5',
'lh: 5-7',
'lh: 7-10',
'lh: 10-',
'p4: 0-0.5',
'p4: 0.5-1',
'p4: 1-1.5',
'p4: 1.5-2',
'p4: 2-',
'bhcg: 0-5',
'bhcg: 5-',
'sono: quiet',
'afs: 1-5',
'cyst: > 15',
'afs: 6-10',
'afs: 10-20',
'afs: 20+',
'sono: >15: <35% and >8: >35%',
'sono: >15: >35% and >8: >35%',
'sono: >20: 0-15% and >15: <35%',
'sono: >20: 0-15% and >15: >35%',
'sono: >20: >15% and >15: <35%',
'sono: >20: >15% and >15: >35%',
'sono: ovulate',
'endo: <5',
'endo: 5-7',
'endo: 7-13',
'endo: 13-',
'folli: 75.0',
'folli: 75.0 eod',
'folli: 150.0',
'folli: 150.0 eod',
'folli: 225.0',
'folli: 225.0 eod',
'folli partially missed',
'folli: no dose',
'clomid: 50.0',
'clomid: 100.0',
'clomid partially missed',
'clomid: no dose',
'fem: 2.5',
'fem: 5.0',
'fem partially missed',
'fem: no dose',
'cetrorelix: 1/3 syringe',
'cetrorelix: 1/2 syringe',
'cetrorelix partially missed',
'cetrorelix: no dose',
'ocp: 0.5',
'ocp: 0.5 eod',
'ocp: 1.0',
'ocp: 1.0 eod',
'ocp partially missed',
'ocp: no dose',
'est: 1.0',
'est: 2.0',
'est: 4.0',
'est: 6.0',
'est partially missed',
'est: no dose',
'ovidrel',
'lupron',
'no trigger',
'cycle day: 0-3',
'cycle day: 4-7',
'cycle day: 8-11',
'cycle day: 12-15',
'cycle day: 16-19',
'cycle day: 20-23',
'cycle day: 24-27',
'cycle day: 28-',
]

label_index = [BertVocab['[CLS]'], 
                BertVocab['[SEP]'], 
                BertVocab['fsh: 0-5'], 
                BertVocab['fsh: 5-15'], 
                BertVocab['fsh: 15-30'],
                BertVocab['fsh: 30-40'],
                BertVocab['fsh: 40-'],
                BertVocab['e2: 0-50'], 
                BertVocab['e2: 50-100'], 
                BertVocab['e2: 100-200'],
                BertVocab['e2: 200-500'],
                BertVocab['e2: 500-1000'], 
                BertVocab['e2: 1000-1500'], 
                BertVocab['e2: 1500-2000'],
                BertVocab['e2: 2000-3000'],
                BertVocab['e2: 3000-'],
                BertVocab['lh: 0-1'], 
                BertVocab['lh: 1-1.5'], 
                BertVocab['lh: 1.5-2'],
                BertVocab['lh: 2-3'], 
                BertVocab['lh: 3-5'], 
                BertVocab['lh: 5-7'],
                BertVocab['lh: 7-10'],
                BertVocab['lh: 10-'],
                BertVocab['p4: 0-0.5'], 
                BertVocab['p4: 0.5-1'], 
                BertVocab['p4: 1-1.5'],
                BertVocab['p4: 1.5-2'], 
                BertVocab['p4: 2-'], 
                BertVocab['bhcg: 0-5'],
                BertVocab['bhcg: 5-'],
                BertVocab['sono: quiet'],
                BertVocab['afs: 1-5'],
                BertVocab['cyst: > 15'],
                BertVocab['afs: 6-10'], 
                BertVocab['afs: 10-20'],
                BertVocab['afs: 20+'], 
                BertVocab['sono: >15: <35% and >8: >35%'], 
                BertVocab['sono: >15: >35% and >8: >35%'],
                BertVocab['sono: >20: 0-15% and >15: <35%'],
                BertVocab['sono: >20: 0-15% and >15: >35%'],
                BertVocab['sono: >20: >15% and >15: <35%'], 
                BertVocab['sono: >20: >15% and >15: >35%'], 
                BertVocab['sono: ovulate'],
                BertVocab['endo: <5'], 
                BertVocab['endo: 5-7'], 
                BertVocab['endo: 7-13'],
                BertVocab['endo: 13-'],
                BertVocab['folli: 75.0'],
                BertVocab['folli: 75.0 eod'],
                BertVocab['folli: 150.0'], 
                BertVocab['folli: 150.0 eod'], 
                BertVocab['folli: 225.0'],
                BertVocab['folli: 225.0 eod'],
                BertVocab['folli partially missed'],
                BertVocab['folli: no dose'],
                BertVocab['clomid: 50.0'],
                BertVocab['clomid: 100.0'],
                BertVocab['clomid partially missed'], 
                BertVocab['clomid: no dose'],
                BertVocab['fem: 2.5'], 
                BertVocab['fem: 5.0'],
                BertVocab['fem partially missed'],
                BertVocab['fem: no dose'],
                BertVocab['orli: 0.125 eod'], 
                BertVocab['orli: 0.25 eod'], 
                BertVocab['orli partially missed'], 
                BertVocab['orli: no dose'],
                BertVocab['ocp: 0.5'],
                BertVocab['ocp: 0.5 eod'],
                BertVocab['ocp: 1.0'],
                BertVocab['ocp: 1.0 eod'], 
                BertVocab['ocp partially missed'], 
                BertVocab['ocp: no dose'],
                BertVocab['est: 1.0'],
                BertVocab['est: 2.0'],
                BertVocab['est: 4.0'], 
                BertVocab['est: 6.0'], 
                BertVocab['est partially missed'],
                BertVocab['est: no dose'],
                BertVocab['ovidrel'],
                BertVocab['lupron'],
                BertVocab['no trigger'],
                BertVocab['cycle day: 0-3'], 
                BertVocab['cycle day: 4-7'], 
                BertVocab['cycle day: 8-11'],
                BertVocab['cycle day: 12-15'],
                BertVocab['cycle day: 16-19'],
                BertVocab['cycle day: 20-23'],
                BertVocab['cycle day: 24-27'],
                BertVocab['cycle day: 28-'],
                ]
w = edwards_model.bert.embeddings.word_embeddings.weight[label_index]

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(w.data.cpu().numpy())
# Two dimensions for each of our images
 
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label':label})
fig, ax = plt.subplots(1, figsize=(10,10))
plt.rcParams["lines.markeredgewidth"] = 1
markers = {
 '[CLS]': '.',
 '[SEP]': '.',
 'fsh: 0-5': 'p',
 'fsh: 5-15': 'p',
 'fsh: 15-30': 'p',
 'fsh: 30-40': 'p',
 'fsh: 40-': 'p',
 'e2: 0-50': 'P',
 'e2: 50-100': 'P',
 'e2: 100-200': 'P',
 'e2: 200-500': 'P',
 'e2: 500-1000': 'P',
 'e2: 1000-1500': 'P',
 'e2: 1500-2000': 'P',
 'e2: 2000-3000': 'P',
 'e2: 3000-': 'P',
 'lh: 0-1': '*',
 'lh: 1-1.5': '*',
 'lh: 1.5-2': '*',
 'lh: 2-3': '*',
 'lh: 3-5': '*',
 'lh: 5-7': '*',
 'lh: 7-10': '*',
 'lh: 10-': '*',
 'p4: 0-0.5': 'X',
 'p4: 0.5-1': 'X',
 'p4: 1-1.5': 'X',
 'p4: 1.5-2': 'X',
 'p4: 2-': 'X',
 'bhcg: 0-5': 'D',
 'bhcg: 5-': 'D',
 'sono: quiet': 'd',
 'sono: ovulate': 'd',
 'sono: >20: >15% and >15: >35%': 'd',
 'sono: >20: >15% and >15: <35%': 'd',
 'sono: >20: 0-15% and >15: >35%': 'd',
 'sono: >20: 0-15% and >15: <35%': 'd',
 'sono: >15: >35% and >8: >35%': 'd',
 'sono: >15: <35% and >8: >35%': 'd',
 'afs: 1-5': '8',
 'afs: 6-10': '8',
 'afs: 10-20': '8',
 'afs: 20+': '8',
 'endo: <5': 'H',
 'endo: 5-7': 'H',
 'endo: 7-13': 'H',
 'endo: 13-': 'H',
 'cyst: > 15': 'H',
 'folli: 75.0': 'v',
 'folli: 75.0 eod': 'v',
 'folli: 150.0': 'v',
 'folli: 150.0 eod': 'v',
 'folli: 225.0': 'v',
 'folli: 225.0 eod': 'v',
 'folli: no dose': 'v',
 'folli partially missed': 'v',
 'clomid: 50.0': 'v',
 'clomid: 100.0': 'v',
 'clomid: no dose': 'v',
 'clomid partially missed': 'v',
 'fem: 2.5': 'v',
 'fem: 5.0': 'v',
 'fem: no dose': 'v',
 'fem partially missed': 'v',
 'cetrorelix: 1/3 syringe' : '>',
 'cetrorelix: 1/2 syringe' : '>',
 'cetrorelix partially missed' : '>',
 'cetrorelix: no dose' : '>',
 'ocp: 0.5': '<',
 'ocp: 0.5 eod': '<',
 'ocp: 1.0': '<',
 'ocp: 1.0 eod': '<',
 'ocp: no dose': '<',
 'ocp partially missed': '<',
 'est: 1.0': '<',
 'est: 2.0': '<',
 'est: 4.0': '<',
 'est: 6.0': '<',
 'est: no dose': '<',
 'est partially missed': '<',
 'ovidrel': '^',
 'lupron': '^',
 'no trigger': '^',
 'cycle day: 0-3': 'o',
 'cycle day: 4-7': 'o',
 'cycle day: 8-11': 'o',
 'cycle day: 12-15': 'o',
 'cycle day: 16-19': 'o',
 'cycle day: 20-23': 'o',
 'cycle day: 24-27': 'o',
 'cycle day: 28-': 'o',
 }
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=100, style='label', markers=markers)
lim = (tsne_result.min()-2, tsne_result.max()+2)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, ncols=3)

