
import sys 
sys.path.insert(0, '../../')

import time
from datetime import datetime as dt
import os,io


import numpy as np
import pandas as pd
import random

from transformers import BertConfig
from common.common import create_folder
from common.pytorch import load_model
from dataLoader.utils import gen_vocab
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from preprocess.standard_config import * 
from dataLoader.Loader_for_test_downstream_classifier import MLMLoader

from torch import nn, argmax, flatten
from model.mlm import EdwardsForMultiLabelPrediction

from model.optimiser import adam
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, top_k_accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

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

train_params = {
    'batch_size': 10,
    'use_cuda': True,
    'max_len_seq': 128,
    'device': 'cuda:0',
    'type': 'e2'
}

global_params = {
    'max_seq_len': train_params['max_len_seq'],
    'min_visit': 2,
    'gradient_accumulation_steps': 1,
    'output_dir': './sessions/%s_downstream/' % train_params['type'], # output folder
    'best_name': '%s_classifier' % train_params['type'],  # output model name
    'device': 'cuda:0'
}

optim_config = {
    'lr': 1e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

file_config = {
    'core_vocab':'',  # vocabulary idx2token, token2idx
    'traindata': './data/data_w_date_new_split_w_profiles_v5.csv',  # formated data 
    'testdata': './data/data_test_w_date_split_w_profiles_v3.csv',  # formated data 
    'model_path': './sessions/', # where to save model
    'model_name': 'edwards', # model name
    'classifier_name': '%s_classifier' % train_params['type'],
    'file_name': 'edwards_%s_classifier.log'  % train_params['type'],  # log path
}
create_folder(file_config['model_path'])

sig = nn.Sigmoid()

BertVocab = gen_vocab(CORE_EMBEDDING_CODE_VOC)
BertDecoder = dict((v,k) for k,v in BertVocab.items() )
DemoVocab = gen_vocab(DEMOGRAFIC_EMBEDDING_CODE_VOC)

traindata = pd.read_csv(file_config['traindata'])
testdata = pd.read_csv(file_config['testdata'])
traindata = traindata.loc[traindata['demo_text'].isna() == False]
testdata = testdata.loc[testdata['demo_text'].isna() == False]

# remove patients with visits less than min visit
traindata['length'] = traindata['core_text'].apply(lambda x: len([i for i in range(len(x.split(', '))) if x.split(', ')[i] == '[SEP]']))
testdata['length'] = testdata['core_text'].apply(lambda x: len([i for i in range(len(x.split(', '))) if x.split(', ')[i] == '[SEP]']))

traindata = traindata.loc[traindata['length'] >= global_params['min_visit']]
traindata = traindata.reset_index(drop=True)
testdata = testdata.loc[testdata['length'] >= global_params['min_visit']]
testdata = testdata.reset_index(drop=True)

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

def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def precision(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    label, output=label.cpu(), output.detach().cpu()
    tempprc= average_precision_score(label.numpy(),output.numpy(),average='samples')
    return tempprc, output, label


def precision_test(logits, one_hot, label):
    softmax = nn.Softmax(dim=1)
    sig = nn.Sigmoid()
    test_label_set = list(set(label.cpu().numpy()))
    output=sig(logits)
    logits = logits[:, test_label_set]
    label, output, logits, one_hot =label.cpu(), output.detach().cpu(), softmax(logits).cpu(), one_hot.cpu()
    tempprc= average_precision_score(one_hot.numpy(),output.numpy(),average='samples')
    roc = roc_auc_score(label.numpy(), logits.numpy(), multi_class='ovr',labels=test_label_set)
    top_2_accuracy_score = top_k_accuracy_score(label.numpy(), logits.numpy(), k=2)
    return tempprc, roc, output, label, top_2_accuracy_score

def train(
    net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device, f: io.TextIOWrapper, e: int, fold: int,
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
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0

    y = []
    y_label = []
    logits_all = []
    
    data_loader_iter = iter(data_loader)
    _print_idx = data_loader.__len__() // 10
    for idx in range(data_loader.__len__()):
        cnt += 1
        if idx % _print_idx == 0:
            print('going %s till to %s ...' % (idx+1, data_loader.__len__()))

        batch  = next(data_loader_iter)
        batch = tuple(t.to(train_params['device']) for t in batch)
        age_ids, bmi_ids, cycle_len_ids, input_ids, posi_ids, segment_ids, attMask, masked_label, artcycle_ids, output_pred_masks = batch
        onehot_labels = nn.functional.one_hot(masked_label[:, 0], num_classes=len(CODE_SUB_GROUP_W_KEYS[train_params['type']]))


        loss, logits = net(input_ids, age_ids=age_ids, bmi_ids=bmi_ids, 
                            cycle_len_ids=cycle_len_ids, seg_ids=segment_ids, 
                            posi_ids=posi_ids, attention_mask=attMask, 
                            masked_lm_labels=onehot_labels, artcycle_ids=artcycle_ids)
        


        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        y_label.append(flatten(masked_label[:, 0]).cpu())

        y.append(argmax(logits, dim=1).cpu())
        logits_all.append(logits.detach().cpu())

        if idx % 50==0 and idx > 0:
            prec, a, b = precision(logits, onehot_labels)
            y_label = torch.cat(y_label, dim=0)
            y = torch.cat(y, dim=0)
            cm = confusion_matrix(y_label, y)
            logits_all = torch.cat(logits_all, dim=0)

            f.write("fold: {} \t| epoch: {}\t| Cnt: {}\t| Loss: {}\t| precision: {}\n".format(fold, e, cnt,temp_loss/500, prec))
            print("fold: {} \t| epoch: {}\t| Cnt: {}\t| Loss: {}\t| precision: {}".format(fold, e, cnt,temp_loss/500, prec))
            f.write("fold: {} \t| epoch: {}\t| Cnt: {}\t| CM: \n{}\n".format(fold, e, cnt, cm))
            print("fold: {} \t| epoch: {}\t| Cnt: {}\t| CM: \n{}\n".format(fold, e, cnt, cm))
            
            temp_loss = 0
            y_label = []
            y = []
            logits_all =[]
        
        if (idx + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

def evaluation(net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device, f: io.TextIOWrapper, e: int, fold: int,
) -> float:
    net.eval()
    y = []
    y_label = []
    tr_loss = 0
    logits_all = []
    onehot_all = []
    data_loader_iter = iter(data_loader)
    
    for idx in range(data_loader.__len__()):
        batch  = next(data_loader_iter)
        batch = tuple(t.to(train_params['device']) for t in batch)

        age_ids, bmi_ids, cycle_len_ids, input_ids, posi_ids, segment_ids, attMask, masked_label, artcycle_ids, output_pred_masks = batch
        onehot_labels = nn.functional.one_hot(masked_label[:, 0], num_classes=len(CODE_SUB_GROUP_W_KEYS[train_params['type']]))

        with torch.no_grad():
            loss, logits = net(input_ids, age_ids=age_ids, bmi_ids=bmi_ids, 
                            cycle_len_ids=cycle_len_ids, seg_ids=segment_ids, 
                            posi_ids=posi_ids, attention_mask=attMask, 
                            masked_lm_labels=onehot_labels, artcycle_ids=artcycle_ids)

            tr_loss += loss.item()

            y_label.append(flatten(masked_label[:, 0]).cpu())
            y.append(argmax(sig(logits), dim=1).cpu())

            logits_all.append(logits.cpu())
            onehot_all.append(onehot_labels.cpu())


    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)

    logits_all = torch.cat(logits_all, dim=0)
    onehot_all = torch.cat(onehot_all, dim=0)

    aps, roc, output, label, top2 = precision_test(logits_all, onehot_all, y_label)

    cm = confusion_matrix(y_label, y)
    return aps, roc, top2, tr_loss, cm


Tset = MLMLoader(testdata, BertVocab, DemoVocab, max_len=train_params['max_len_seq'], result=train_params['type'])

f = open(os.path.join(file_config['model_path'], file_config['file_name']), "w")
best_pre, best_test_pre = 0.0, 0.0
k=5

test_loader = DataLoader(Tset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)
traindata = traindata.loc[traindata[train_params['type']].isna()==False].reset_index(drop=True)

for fold in range(k):
    f.write('Fold {} \n'.format(fold + 1))
    print('Fold {}'.format(fold + 1))
    _labels, _counts = np.unique(traindata[train_params['type']], return_counts=True)

    label_weights = {k:1 for k in _labels}
    label_count = sum(_counts)

    for k,v in zip(_labels,_counts):
        if v > 0:
            label_weights[k] = label_count/v

    sample_weights = [label_weights[k] for k in traindata[train_params['type']]]
    assert len(sample_weights) == label_count

    print([label_weights[k] for k in sorted(label_weights.keys())])

    sampler = WeightedRandomSampler(sample_weights, int(label_count))

    Dset = MLMLoader(traindata, BertVocab, DemoVocab, max_len=train_params['max_len_seq'], result=train_params['type'])
    train_loader = DataLoader(Dset, batch_size=train_params['batch_size'], sampler=sampler, num_workers=3, drop_last=True)


    conf = EdwardsConfig(model_config)

    artbert_model = EdwardsForMultiLabelPrediction(conf, len(CODE_SUB_GROUP_W_KEYS[train_params['type']]))


    output_model_file = os.path.join(file_config['model_path'], file_config['model_name'])
    artbert_model = load_model(output_model_file, artbert_model)
    artbert_model = artbert_model.to(global_params['device'])

    optim = adam(params=list(artbert_model.named_parameters()), config=optim_config)
    
    for i in range(3):
        f.write('start fold %s #%s round training...\n' % (fold,i))
        print('start fold %s #%s round training...' % (fold,i))
        train(net=artbert_model,
                data_loader=train_loader,
                dtype=torch.float,
                device=train_params['device'],
                f=f,
                e=i,
                fold=fold)

        f.write('start fold %s #%s round verification...\n' % (fold,i))
        print('start fold %s #%s round verification...' % (fold,i))
        test_aps, test_roc, test_top2, test_loss, test_cm = evaluation(net=artbert_model,
                data_loader=test_loader,
                dtype=torch.float,
                device=train_params['device'],
                f=f,
                e=i,
                fold=fold)

        if test_aps > best_test_pre:
            # Save a trained model
            f.write("** ** * Saving fine - tuned model fold %s #%s round ** ** * \n" % (fold, i))
            print("** ** * Saving fine - tuned model fold %s #%s round ** ** * " % (fold, i))
            model_to_save = artbert_model.module if hasattr(artbert_model, 'module') else artbert_model  # Only save the model it-self
            output_model_file = os.path.join(global_params['output_dir'],global_params['best_name'])
            create_folder(global_params['output_dir'])

            torch.save(model_to_save.state_dict(), output_model_file)
            best_test_pre = test_aps

        f.write('test aps : {}\n\n'.format(test_aps))
        print('test aps : {}'.format(test_aps))
        f.write('test roc : {}\n\n'.format(test_roc))
        print('test roc : {}'.format(test_roc))
        f.write('test top2 : {}\n\n'.format(test_top2))
        print('test top2 : {}'.format(test_top2))
        f.write('test cm : \n{}\n\n'.format(test_cm))
        print('test cm : \n{}'.format(test_cm))

f.close()


Tset = MLMLoader(testdata, BertVocab, DemoVocab, max_len=train_params['max_len_seq'], result=train_params['type'])

f = open(os.path.join(file_config['model_path'], file_config['file_name']), "w")
best_pre, best_test_pre = 0.0, 0.0
k=5

test_loader = DataLoader(Tset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)
traindata = traindata.loc[traindata[train_params['type']].isna()==False].reset_index(drop=True)

for fold in range(k):
    f.write('Fold {} \n'.format(fold + 1))
    print('Fold {}'.format(fold + 1))
    _labels, _counts = np.unique(traindata[train_params['type']], return_counts=True)

    label_weights = {k:1 for k in _labels}
    label_count = sum(_counts)
    for k,v in zip(_labels,_counts):
        if v > 0:
            label_weights[k] = label_count/v

    sample_weights = [label_weights[k] for k in traindata[train_params['type']]]
    assert len(sample_weights) == label_count

    print([label_weights[k] for k in sorted(label_weights.keys())])

    sampler = WeightedRandomSampler(sample_weights, int(label_count))

    Dset = MLMLoader(traindata, BertVocab, DemoVocab, max_len=train_params['max_len_seq'], result=train_params['type'])
    train_loader = DataLoader(Dset, batch_size=train_params['batch_size'], sampler=sampler, num_workers=3, drop_last=True)


    conf = EdwardsConfig(model_config)
    artbert_model = EdwardsForMultiLabelPrediction(conf, len(CODE_SUB_GROUP_W_KEYS[train_params['type']]))

    output_model_file = os.path.join(file_config['model_path'], file_config['model_name'])
    artbert_model = load_model(output_model_file, artbert_model)
    artbert_model = artbert_model.to(global_params['device'])

    optim = adam(params=list(artbert_model.named_parameters()), config=optim_config)
    
    for i in range(3):
        f.write('start fold %s #%s round training...\n' % (fold,i))
        print('start fold %s #%s round training...' % (fold,i))
        train(net=artbert_model,
                data_loader=train_loader,
                dtype=torch.float,
                device=train_params['device'],
                f=f,
                e=i,
                fold=fold)

        f.write('start fold %s #%s round verification...\n' % (fold,i))
        print('start fold %s #%s round verification...' % (fold,i))
        test_aps, test_roc, test_top2, test_loss, test_cm = evaluation(net=artbert_model,
                data_loader=test_loader,
                dtype=torch.float,
                device=train_params['device'],
                f=f,
                e=i,
                fold=fold)

        if test_aps > best_test_pre:
            # Save a trained model
            f.write("** ** * Saving fine - tuned model fold %s #%s round ** ** * \n" % (fold, i))
            print("** ** * Saving fine - tuned model fold %s #%s round ** ** * " % (fold, i))
            model_to_save = artbert_model.module if hasattr(artbert_model, 'module') else artbert_model  # Only save the model it-self
            output_model_file = os.path.join(global_params['output_dir'],global_params['best_name'])
            create_folder(global_params['output_dir'])

            torch.save(model_to_save.state_dict(), output_model_file)
            best_test_pre = test_aps

        f.write('test aps : {}\n\n'.format(test_aps))
        print('test aps : {}'.format(test_aps))
        f.write('test roc : {}\n\n'.format(test_roc))
        print('test roc : {}'.format(test_roc))
        f.write('test top2 : {}\n\n'.format(test_top2))
        print('test top2 : {}'.format(test_top2))
        f.write('test cm : \n{}\n\n'.format(test_cm))
        print('test cm : \n{}'.format(test_cm))

f.close()


