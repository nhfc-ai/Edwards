from torch.utils.data.dataset import Dataset
import numpy as np
import statistics
from sklearn import preprocessing
import pandas as pd
from datetime import datetime as dt
from dataLoader.utils import seq_padding,position_idx,index_seg,\
                    random_mask,gen_vocab,token2idx,mask_by_type,\
                    mask_by_type_last_point,mask_by_type_last_point_for_next_visit, \
                    pack_stream_for_deploy, pack_stream_for_seq2seq, \
                    wrap_wo_mask_downstream, convert_label_starting_with_zero
import torch
from preprocess.standard_config import * 

class MLMLoader(Dataset):
    def __init__(self, dataframe, core_vocab, demo_vocab, encoder_max_len, decoder_max_len, core_col='core_text', demo_col='demo_text', result='final_result'):
        # self.dataframe = dataframe.loc[dataframe[core_col].isna() == False]

        if result in ['final_result', 'twopn_label', 'mii_label','blst_label']:
            self.dataframe = dataframe.loc[dataframe[core_col].isna() == False].\
                                loc[(dataframe[core_col].str.contains(r'lupron')) | \
                                (dataframe[core_col].str.contains(r'ovidrel')) | \
                                (dataframe[core_col].str.contains(r'no trigger'))].reset_index(drop=True)
        else:
            self.dataframe = dataframe.loc[dataframe[core_col].isna() == False].\
                                loc[(dataframe[result].isna() == False)].reset_index(drop=True)
        self.core_vocab = core_vocab
        self.demo_vocab = demo_vocab
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.core = self.dataframe[core_col]
        self.demo = self.dataframe[demo_col]
        self.start_date = self.dataframe['date']
        self.cid = self.dataframe['artcycle_id']
        self.type = result
        self.result = self.dataframe[result]
        
        
    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        le = preprocessing.LabelEncoder()
        le.fit(list(self.cid)+[''])

        full_core = self.core[index].split(', ')
        

        if self.type in ['final_result', 'twopn_label', 'mii_label','blst_label']:
            result = self.result[index]
            _trigger_idx = max(self.find_trigger_index(full_core, 'lupron'), 
                                self.find_trigger_index(full_core, 'ovidrel'), 
                                self.find_trigger_index(full_core, 'no trigger'),
                                self.find_trigger_index(full_core, 'ovidrel+lupron'))
            
            _end_idx = full_core[_trigger_idx:].index('[SEP]')+_trigger_idx+1


            if _end_idx > self.encoder_max_len:
                core = full_core[(_end_idx-self.encoder_max_len+1):_end_idx+1]
            else:
                core = full_core[:min(len(full_core), self.encoder_max_len)]
            
        else:
            core = full_core[(-self.encoder_max_len+7):]
            result = convert_label_starting_with_zero(self.type, self.result[index])
        
        # cid = self.cid[index][:self.cid[index].index(' ')]
        cid = self.cid[index]
        demo = self.demo[index].split(', ')
        assert len(demo) == 3
        demo = token2idx(demo, self.demo_vocab)


        # # extract data
        # age = self.age[index][(-self.max_len+1):]
        # code = self.code[index][(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if core[0] != '[CLS]' and core[0] != '[SEP]':
            core[0] = '[CLS]'

        if core[0] == '[SEP]':
            core.pop(0)

        if core[-1] != '[SEP]':
            core[-1] = '[SEP]'

        if core[-1] == core[-2] == '[SEP]':
            core.pop(-1)

        # demo_len = min(len(full_core)+7, self.max_len)

        # demo = np.array([demo]*demo_len).transpose()
        
        # if demo_len < self.max_len:
        #     demo = np.pad(demo, ((0, 0), (0, self.max_len-demo_len)), 'constant', constant_values=self.demo_vocab['[PAD]'])

        # age, bmi, cycle_len = demo


        _start_date_obj = dt.strptime(self.start_date[index], '%Y-%m-%d')
        # if self.mask_domain == 'next_visit':
        #     tokens, core, label, artcycle_ids = mask_by_type_last_point_for_next_visit(core, self.core_vocab, mask_token_ids=mask_list, artcycle_ids=cid)
        # else:
        #     tokens, core, label, artcycle_ids = mask_by_type_last_point(core, self.core_vocab, mask_token_ids=mask_list, artcycle_ids=cid)

        tokens, encoder_core, encoder_label, encoder_artcycle_ids, decoder_core, decoder_label, decoder_artcycle_ids, classifier_label = pack_stream_for_seq2seq(core, self.core_vocab, cid, mode=self.type, result=result)
        # print(classifier_label)
        # input()

        # assert classifier_label is not None
        # if classifier_label is None:
        #     result = 0
        # elif :
        #     result = convert_label_starting_with_zero(self.type, classifier_label)
        # print(tokens)
        # print('\n')
        # print(encoder_core)
        # print(encoder_label)
        # print(encoder_artcycle_ids)
        # print('\n')
        # print(decoder_core)
        # print(decoder_label)
        # print(decoder_artcycle_ids)
        # print('\n')
        # print(demo)
        # input()
        # get position code and segment code
        tokens = seq_padding(tokens, self.encoder_max_len)
        # position = position_idx(tokens)
        # segment = index_seg(tokens)

        en_demo_len = min(len(encoder_core)+7, self.encoder_max_len)

        en_demo = np.array([demo]*en_demo_len).transpose()
        
        # pad age sequence and code sequence
        if en_demo_len < self.encoder_max_len:
            en_demo = np.pad(en_demo, ((0, 0), (0, self.encoder_max_len-en_demo_len)), 'constant', constant_values=self.demo_vocab['[PAD]'])

        en_age, en_bmi, en_cycle_len = en_demo


        de_demo_len = min(len(decoder_core)+7, self.decoder_max_len)

        de_demo = np.array([demo]*de_demo_len).transpose()
        
        # pad age sequence and code sequence
        if de_demo_len < self.decoder_max_len:
            de_demo = np.pad(de_demo, ((0, 0), (0, self.decoder_max_len-de_demo_len)), 'constant', constant_values=self.demo_vocab['[PAD]'])

        de_age, de_bmi, de_cycle_len = de_demo

        # pad code and label
        encoder_core = seq_padding(encoder_core, self.encoder_max_len, symbol=self.core_vocab['[PAD]'])
        encoder_label = seq_padding(encoder_label, self.encoder_max_len, symbol=-1)
        encoder_artcycle_ids = seq_padding(encoder_artcycle_ids, self.encoder_max_len, symbol='')
        self.encoder_codes = encoder_core

        decoder_core = seq_padding(decoder_core, self.decoder_max_len, symbol=self.core_vocab['[PAD]'])
        decoder_label = seq_padding(decoder_label, self.decoder_max_len, symbol=-1)
        decoder_artcycle_ids = seq_padding(decoder_artcycle_ids, self.decoder_max_len, symbol='')
        classifier_label = seq_padding(classifier_label, self.decoder_max_len, symbol=0)
        self.decoder_codes = decoder_core

        # if self.type in ['final_result', 'twopn_label', 'mii_label','blst_label']:
        #     _, _, label, _, _  = wrap_wo_mask_downstream(core, self.core_vocab,  artcycle_ids=cid,  result=result, mode=self.type)
        # else:
        #     # mask_list = BLOOD_TEST_CODE + SONO_TEST_CODE + TREATMENT_CODE
        #     # tokens, core, _, artcycle_ids  = mask_by_type_last_point_for_next_visit(core, self.core_vocab, mask_token_ids=mask_list, artcycle_ids=cid)

        #     # output_pred_masks = [0]*len(tokens)
        #     label = [result]*len(tokens)

        # print(index)
        # print(self.core[index])
        # print(core)
        # print(label)
        # print(self.demo[index])
        # print(de_age)
        # print(de_bmi)
        # print(de_cycle_len)
        # print(len(label))
        # print(len(artcycle_ids))
        # if len(label) != len(artcycle_ids):
        #     print(label)
        #     print(artcycle_ids)
        #     print(len(label))
        #     print(len(artcycle_ids))
        # print([len(core), len(label), len(artcycle_ids)])
        # input()

        return torch.LongTensor(en_age), torch.LongTensor(en_bmi), torch.LongTensor(en_cycle_len), \
                torch.LongTensor(encoder_core), torch.LongTensor(encoder_label), \
                torch.LongTensor(le.transform(encoder_artcycle_ids)), \
                torch.LongTensor(de_age), torch.LongTensor(de_bmi), torch.LongTensor(de_cycle_len), \
                torch.LongTensor(decoder_core), \
                torch.LongTensor(decoder_label), torch.LongTensor(le.transform(decoder_artcycle_ids)), \
                torch.LongTensor(classifier_label)

    def __len__(self):
        return len(self.core)

    def __tokens__(self, index):
        return self.core[index]

    def __codes__(self, index):
        _token_list = self.core[index].split(', ')
        return token2idx(_token_list, self.core_vocab)

    def __cid__(self, index):
        return self.cid[index]
    
    def find_trigger_index(self, l, kw):
        try:
            return l.index(kw)
        except ValueError:
            return -1

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