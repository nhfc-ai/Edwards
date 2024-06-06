
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
from dataLoader.utils import gen_vocab
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from preprocess.standard_config import * 
from dataLoader.MLM_stim_starter import MLMLoader

from torch import nn
from model.mlm import EdwardsForMaskedLM

from model.optimiser import adam
import sklearn.metrics as skm

from sklearn import preprocessing
cid_encoder = preprocessing.LabelEncoder()
sig = nn.Sigmoid()

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
    'testdata': './data/data_test_w_date_new_v3.csv', #test data
    'model_path': './sessions/', # where to save model
    'model_name': 'edwards', # model name
    'file_name': 'training.log',  # log path
    'file_name_detail': 'training_detail.log',  # log path
}
create_folder(file_config['model_path'])

global_params = {
    'max_seq_len': 128,
    'min_visit': 2,
    'gradient_accumulation_steps': 1
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
    'device': 'cuda:0'
}

BertVocab = gen_vocab(CORE_EMBEDDING_CODE_VOC)
BertDecoder = dict((v,k) for k,v in BertVocab.items() )
DemoVocab = gen_vocab(DEMOGRAFIC_EMBEDDING_CODE_VOC)

data = pd.read_csv(file_config['data'])
data = data.loc[(data['core_text'].isna() == False) & (data['demo_text'].isna() == False)]
# remove patients with visits less than min visit
data['length'] = data['core_text'].apply(lambda x: len([i for i in range(len(x.split(', '))) if x.split(', ')[i] == '[SEP]']))
data = data.loc[data['length'] >= global_params['min_visit']]
data = data.reset_index(drop=True)

testdata = pd.read_csv(file_config['testdata'])
testdata = testdata.loc[(testdata['core_text'].isna() == False) & (testdata['demo_text'].isna() == False)]

# remove patients with visits less than min visit
testdata['length'] = testdata['core_text'].apply(lambda x: len([i for i in range(len(x.split(', '))) if x.split(', ')[i] == '[SEP]']))
testdata = testdata.loc[testdata['length'] >= global_params['min_visit']]
testdata = testdata.reset_index(drop=True)

Dset = MLMLoader(data, BertVocab, DemoVocab, max_len=train_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)
cid_encoder.fit(list(Dset.cid) + [''])

Tset = MLMLoader(testdata, BertVocab, DemoVocab, max_len=train_params['max_len_seq'])
testload = DataLoader(dataset=Tset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)

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
model = EdwardsForMaskedLM(conf)
model.set_output_embeddings_weight(model.bert.embeddings.word_embeddings.weight)

model = model.to(train_params['device'])
optim = adam(params=list(model.named_parameters()), config=optim_param)

def cal_acc(label, pred):
    logs = nn.LogSoftmax(dim=-1)
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.Tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    top_2_accuracy_score = skm.top_k_accuracy_score(truelabel,  truepred.numpy(), k=2, labels=np.array(list(BertVocab.values())))
    return precision, top_2_accuracy_score

def show_pred(label, pred, f):
    logs = nn.LogSoftmax(dim=-1)
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.Tensor(truepred)).numpy()
    truepred_top2_idx = np.argsort(truepred, -1)[:,::-1][:,:2]
    truepred_top2_prob = np.take_along_axis(truepred, truepred_top2_idx, -1)
    for i, token in enumerate(truelabel):
        top1_idx, top2_idx = truepred_top2_idx[i]
        top1_prob, top2_prob = truepred_top2_prob[i]
        f.write('label: %s, top1: %s, top2: %s, top1 prb: %s, top2 prb %s\n' % 
                (BertDecoder[token], BertDecoder[top1_idx], 
                BertDecoder[top2_idx], top1_prob, top2_prob))

def train(e, loader, max_p, max_top2):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt= 0
    start = time.time()
    local_max_top, local_max_p = [], []

    for step, batch in enumerate(loader):
        cnt +=1
        batch = tuple(t.to(train_params['device']) for t in batch)
        age_ids, bmi_ids, cycle_len_ids, input_ids, posi_ids, segment_ids, attMask, masked_label, artcycle_ids = batch

        loss, pred, label, _ = model(input_ids, age_ids=age_ids, bmi_ids=bmi_ids, 
                            cycle_len_ids=cycle_len_ids, seg_ids=segment_ids, 
                            posi_ids=posi_ids, attention_mask=attMask, 
                            masked_lm_labels=masked_label, artcycle_ids=artcycle_ids)
        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()
        
        temp_loss += loss.item()
        tr_loss += loss.item()
        
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        
        if step % 200==0:
            _p, _top2 =  cal_acc(label, pred)
            local_max_top.append(_top2)
            local_max_p.append(_p)
            print("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| top2: {:.4f}\t| time: {:.2f}".format(e,
                 cnt, temp_loss/2000, _p, _top2, time.time()-start))
            f.write("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| top2: {:.4f}\t| time: {:.2f}".format(e,
                 cnt, temp_loss/2000, _p, _top2, time.time()-start))
            temp_loss = 0
            start = time.time()
            
        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

    local_max_top_avg = sum(local_max_top)/len(local_max_top)
    local_max_p_avg = sum(local_max_p)/len(local_max_p)

    _eval_p, _eval_top2, _eval_label, _eval_pred = evaluate(
        net=model,
        data_loader=testload,
        dtype=torch.float,
        device=train_params['device'],
        f=f,
    )

    print('eval top2 %s current top2 %s avg local top2 %s' % (_eval_top2, max_top2, local_max_top_avg))
    print('eval precision %s current precision %s avg local precision %s' % (_eval_p, max_p, local_max_p_avg))
    f.write('avg local top2 %s current top2 %s avg local top2 %s' % (_eval_top2, max_top2, local_max_top_avg))
    f.write('avg local precision %s current precision %s avg local precision %s\n\n' % (_eval_p, max_p, local_max_p_avg))

    if _eval_p > max_p and _eval_top2 > max_top2:
        print("** ** * Saving fine - tuned model ** ** * ")
        f.write("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        create_folder(file_config['model_path'])
        output_model_file = os.path.join(file_config['model_path'], file_config['model_name'])

        torch.save(model_to_save.state_dict(), output_model_file)
        f2.write("\nepoch: {}\t| cnt: {}\n".format(e, cnt))
        show_pred(_eval_label, _eval_pred, f2)
        
    cost = time.time() - start
    return tr_loss, cost, max(max_p, _eval_p), max(max_top2, _eval_top2)

def evaluate(
    net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device, f: io.TextIOWrapper,
) -> float:
    """
    Compute classification accuracy on provided dataset.
    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()

    y_label = []
    logits_all = []

    _iterator = iter(data_loader)
    with torch.no_grad():
        for idx in range(data_loader.__len__()-1):
            batch = next(_iterator)
            batch = tuple(t.to(train_params['device']) for t in batch)

            age_ids, bmi_ids, cycle_len_ids, input_ids, posi_ids, segment_ids, attMask, masked_label, artcycle_ids = batch
            _, pred, label, _ = net(input_ids, age_ids=age_ids, bmi_ids=bmi_ids, 
                                cycle_len_ids=cycle_len_ids, seg_ids=segment_ids, 
                                posi_ids=posi_ids, attention_mask=attMask, 
                                masked_lm_labels=masked_label, artcycle_ids=artcycle_ids)
            
            y_label.append(label.cpu())

            logits_all.append(pred.cpu())
            
        y_label = torch.cat(y_label, dim=0)
        logits_all = torch.cat(logits_all, dim=0)

        _p, _top2 = cal_acc(y_label, logits_all)
        return _p, _top2, y_label, logits_all

f = open(os.path.join(file_config['model_path'], file_config['file_name']), "w")
f2 = open(os.path.join(file_config['model_path'], file_config['file_name_detail']), "w")
f.write('{}\t{}\t{}\n'.format('epoch', 'loss', 'time'))
max_top2, max_p = 0, 0
for e in range(200):
    loss, time_cost, max_p, max_top2 = train(e, trainload, max_p, max_top2)
    
f.close()
f2.close()    




