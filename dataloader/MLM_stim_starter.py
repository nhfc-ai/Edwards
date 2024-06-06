from torch.utils.data.dataset import Dataset
import numpy as np
import statistics
from sklearn import preprocessing
import pandas as pd
from dataLoader.utils import random_mask_expose_stim_start_w_cid, \
    random_mask_expose_stim_start_w_cid_downgrade_nodose, \
    random_mask_expose_stim_start_w_cid_downgrade_nodose_alt, \
     seq_padding,position_idx,index_seg,random_mask,gen_vocab,token2idx
import torch
from preprocess.standard_config import * 

class MLMLoader(Dataset):
    def __init__(self, dataframe, core_vocab, demo_vocab, max_len, core_col='core_text', demo_col='demo_text'):
        self.core_vocab = core_vocab
        self.demo_vocab = demo_vocab
        self.max_len = max_len
        self.core = dataframe[core_col]
        self.demo = dataframe[demo_col]
        self.cid = dataframe['artcycle_id']
        # print(self.demo)
        
    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        le = preprocessing.LabelEncoder()
        le.fit(list(self.cid)+[''])
        try:
            full_core = self.core[index].split(', ')
            core = full_core[(-self.max_len):]
            demo = self.demo[index].split(', ')
            assert len(demo) == 3
            demo = token2idx(demo, self.demo_vocab)
            cid = self.cid[index]
        except Exception as e:
            print(e)
            print(index)
            print(self.demo[index])
            print(self.core[index])
            pass

        demo_len = min(len(full_core), self.max_len)
        # # extract data
        # age = self.age[index][(-self.max_len+1):]
        # code = self.code[index][(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if core[0] != '[CLS]':
            core[0] = '[CLS]'
        demo = np.array([demo]*demo_len).transpose()
        
        # if core[0] != '[SEP]':
        #     core = np.append(np.array(['[CLS]']), core)
        #     demo = np.array([demo]*demo_len).transpose()
        #     # age = np.append(np.array(age[0]), age)
        # else:
        #     core[0] = '[CLS]'
        #     demo = np.array([demo]*(demo_len-1))

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(core):] = 0

        # pad age sequence and code sequence
        if demo_len < self.max_len:
            demo = np.pad(demo, ((0, 0), (0, self.max_len-demo_len)), 'constant', constant_values=self.demo_vocab['[PAD]'])

        age, bmi, cycle_len = demo

        tokens, core, label, artcycle_ids = random_mask_expose_stim_start_w_cid_downgrade_nodose_alt(core, self.core_vocab, cid)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        core = seq_padding(core, self.max_len, symbol=self.core_vocab['[PAD]'])
        label = seq_padding(label, self.max_len, symbol=-1)
        artcycle_ids = seq_padding(artcycle_ids, self.max_len, symbol='')
        self.codes = core

        # print(index)
        # print(self.core[index])
        # print(core)
        # print(label)
        # print(self.demo[index])
        # print(age)
        # print(bmi)
        # print(cycle_len)
        


        return torch.LongTensor(age), torch.LongTensor(bmi), \
        torch.LongTensor(cycle_len), torch.LongTensor(core), \
         torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), \
                torch.LongTensor(le.transform(artcycle_ids))

    def __len__(self):
        return len(self.core)

    def __tokens__(self, index):
        return self.core[index]

    def __codes__(self, index):
        _token_list = self.core[index].split(', ')
        return token2idx(_token_list, self.core_vocab)

    def __len_stats__(self):
        _len_stats = []
        for _str in self.core:
            _len_stats.append(len(_str.split(', ')))
        return min(_len_stats), max(_len_stats), sum(_len_stats) // len(_len_stats), statistics.median(_len_stats)

    def __visit_stats__(self):
        _visit_stats = []
        for _str in self.core:
            _visit_stats.append(len([c for c in _str.split(', ') if c == '[CLS]']))
        return min(_visit_stats), max(_visit_stats), sum(_visit_stats) // len(_visit_stats), statistics.median(_visit_stats)