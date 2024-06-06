import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F

class Seq2SeqEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(Seq2SeqEmbeddings, self).__init__()
        # self.automodel = AutoModel.from_pretrained("uncased_L-6_H-256_A-4")
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.demographic_embeddings = nn.Embedding(config.demo_vocab_size, config.hidden_size)
        # self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
        #     from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))
        # self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None):
        if age_ids is None:
            age_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        if bmi_ids is None:
            bmi_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        if cycle_len_ids is None:
            cycle_len_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        # if posi_ids is None:
        #     posi_ids = torch.zeros(word_ids.size(), dtype=torch.long)

        core_embed = self.word_embeddings(word_ids)
        age_embed = self.demographic_embeddings(age_ids)
        bmi_embed = self.demographic_embeddings(bmi_ids)
        cycle_len_embed = self.demographic_embeddings(cycle_len_ids)
        # posi_embed = self.posi_embeddings(posi_ids)
        # segment_embed = self.segment_embeddings(seg_ids)

        # if age:
        #     embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        # else:
        #     embeddings = word_embed + segment_embed + posi_embeddings

        embeddings = core_embed + age_embed + bmi_embed + cycle_len_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.dropout_p = config.hidden_dropout_prob

        self.embedding = Seq2SeqEmbeddings(config=config)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, core, age_ids=None, bmi_ids=None, cycle_len_ids=None):
        embedded = self.dropout(self.embedding(core, age_ids, bmi_ids, cycle_len_ids))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, config):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ua = nn.Linear(self.hidden_size, self.hidden_size)
        self.Va = nn.Linear(self.hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, config):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.output_size = config.vocab_size
        self.dropout_p = config.hidden_dropout_prob
        self.embedding = Seq2SeqEmbeddings(config=config)
        self.device = config.device
        self.MAX_OUTPUT_LEN = config.MAX_OUTPUT_LEN
        self.START_TOKEN = config.START_TOKEN

        self.attention = BahdanauAttention(config)
        self.gru = nn.GRU(2 * self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, mode='train', target_tensor=None, 
                age_ids=None, bmi_ids=None, cycle_len_ids=None):
        
        batch_size = encoder_outputs.size(0)
        
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        _iteration = target_tensor.shape[1] if mode == 'train' else self.MAX_OUTPUT_LEN

        decoder_input = self.embedding(target_tensor[:, 0], age_ids[:, 0], bmi_ids[:, 0], cycle_len_ids[:, 0]).unsqueeze(1)
        for i in range(_iteration):
            # print(self.device)
            # print(age_ids)
            # if mode =='train':
            #     decoder_input = self.embedding(target_tensor[:, i], age_ids[:, i], bmi_ids[:, i], cycle_len_ids[:, i]).unsqueeze(1)
            # else:
            #     # _dummy = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.START_TOKEN).to(self.device)
            #     # # print(_dummy)
            #     # decoder_input = self.embedding(_dummy, age_ids, bmi_ids, cycle_len_ids)
                

            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)


            if mode =='train':

                # Teacher forcing: Feed the target as the next input
                decoder_input = self.embedding(target_tensor[:, i], age_ids[:, i], bmi_ids[:, i], cycle_len_ids[:, i]).unsqueeze(1) # Teacher forcing
                # decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = self.embedding(topi.view(-1).detach(), age_ids[:, i], bmi_ids[:, i], cycle_len_ids[:, i]).unsqueeze(1)  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        # print(input)
        # print(input.shape)
        # input()
        embedded =  self.dropout(input)

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    
class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.hidden_size = config.hidden_size
        self.dropout_p = config.hidden_dropout_prob

        self.embedding = Seq2SeqEmbeddings(config=config)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)

        self.encoder = EncoderRNN(config)
        self.decoder = AttnDecoderRNN(config)

        self.criterion = nn.NLLLoss()

        # self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.lr)
        # self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr)

    def forward(self, en_age_ids, en_bmi, en_cycle_day, encoder_core, de_age_ids, de_bmi, de_cycle_day,
                decoder_core, mode='train'):
        encoder_outputs, encoder_hidden = self.encoder(encoder_core, age_ids=en_age_ids, bmi_ids=en_bmi, cycle_len_ids=en_cycle_day)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, mode=mode, target_tensor=decoder_core, 
                age_ids=de_age_ids, bmi_ids=de_bmi, cycle_len_ids=de_cycle_day)

        # print(decoder_outputs.shape)
        # print(decoder_core)
        # input()
        loss = self.criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            decoder_core.view(-1)
        )

        return loss, decoder_outputs.view(-1, decoder_outputs.size(-1))

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        # embedded = self.dropout(self.embedding(input))
        # output, hidden = self.gru(embedded)
        # return output, hidden

class Seq2SeqForMultiLabelPrediction(nn.Module):
    def __init__(self, config, num_labels):
        super(Seq2SeqForMultiLabelPrediction, self).__init__()

        self.hidden_size = config.hidden_size
        self.dropout_p = config.hidden_dropout_prob
        self.num_labels = num_labels

        self.embedding = Seq2SeqEmbeddings(config=config)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)

        self.encoder = EncoderRNN(config)
        self.decoder = AttnDecoderRNN(config)

        self.criterion = nn.NLLLoss()
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, en_age_ids, en_bmi, en_cycle_day, encoder_core, de_age_ids, de_bmi, de_cycle_day,
                decoder_core, label=None, mode='train'):
        # def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None):
        # _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
        #                              output_all_encoded_layers=False)
        assert label is not None
        encoder_outputs, encoder_hidden = self.encoder(encoder_core, age_ids=en_age_ids, bmi_ids=en_bmi, cycle_len_ids=en_cycle_day)
        _, decoder_hidden, _ = self.decoder(encoder_outputs, encoder_hidden, mode=mode, target_tensor=decoder_core, 
                age_ids=de_age_ids, bmi_ids=de_bmi, cycle_len_ids=de_cycle_day)

        # _, pooled_output = self.bert(input_ids, age_ids, bmi_ids, cycle_len_ids, seg_ids, posi_ids, attention_mask,
        #                                output_all_encoded_layers=False)

        pooled_output = self.dropout(decoder_hidden)
        logits = self.classifier(pooled_output)

        loss_fct = nn.MultiLabelSoftMarginLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1, self.num_labels))
        return loss, logits.view(-1, self.num_labels)
