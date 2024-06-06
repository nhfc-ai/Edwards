from torch import nn
from transformers import BertForPreTraining, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertForMaskedLM, BertPredictionHeadTransform
import numpy as np
import torch

class EdwardsEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(EdwardsEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.demographic_embeddings = nn.Embedding(config.demo_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None, seg_ids=None, posi_ids=None):
        if seg_ids is None:
            seg_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        if age_ids is None:
            age_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        if bmi_ids is None:
            bmi_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        if cycle_len_ids is None:
            cycle_len_ids = torch.zeros(word_ids.size(), dtype=torch.long)
        if posi_ids is None:
            posi_ids = torch.zeros(word_ids.size(), dtype=torch.long)

        core_embed = self.word_embeddings(word_ids)
        age_embed = self.demographic_embeddings(age_ids)
        bmi_embed = self.demographic_embeddings(bmi_ids)
        cycle_len_embed = self.demographic_embeddings(cycle_len_ids)
        posi_embed = self.posi_embeddings(posi_ids)
        segment_embed = self.segment_embeddings(seg_ids)

        embeddings = core_embed + segment_embed + posi_embed + age_embed + bmi_embed + cycle_len_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.Tensor(lookup_table)

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # print(hidden_states)
        # print(hidden_states.shape)
        hidden_states = self.transform(hidden_states)
        # print(hidden_states)
        # print(hidden_states.shape)
        # input()
        hidden_states = self.decoder(hidden_states)
        # print(hidden_states)
        # print(hidden_states.shape)
        # input()
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class EdwardsModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super(EdwardsModel, self).__init__(config)
        self.config = config
        self.embeddings = EdwardsEmbeddings(config=config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None, 
                seg_ids=None, posi_ids=None, attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(), dtype=torch.long)
        if age_ids is None:
            age_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        if bmi_ids is None:
            bmi_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        if cycle_len_ids is None:
            cycle_len_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        if seg_ids is None:
            seg_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        if posi_ids is None:
            posi_ids = torch.zeros(input_ids.size(), dtype=torch.long)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, bmi_ids, cycle_len_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_hidden_states=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        if self.pooler:
            pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class EdwardsForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super(EdwardsForMaskedLM, self).__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = EdwardsModel(config, add_pooling_layer=True)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, input_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, masked_lm_labels=None, artcycle_ids=None):
        sequence_output, _ = self.bert(input_ids, age_ids, bmi_ids, cycle_len_ids, seg_ids, posi_ids, attention_mask,
                                       output_all_encoded_layers=False)

        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1), artcycle_ids.view(-1)
        else:
            return prediction_scores, artcycle_ids.view(-1)

    def set_output_embeddings_weight(self, new_embeddings_weight):
        self.cls.predictions.decoder.weight = new_embeddings_weight


class EdwardsForMaskedLMTESTBLASTRATE(BertForMaskedLM):
    def __init__(self, config):
        super(EdwardsForMaskedLMTESTBLASTRATE, self).__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = EdwardsModel(config, add_pooling_layer=True)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, input_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, masked_lm_labels=None, artcycle_ids=None, output_pred_masks=None):
        sequence_output, _ = self.bert(input_ids, age_ids, bmi_ids, cycle_len_ids, seg_ids, posi_ids, attention_mask,
                                       output_all_encoded_layers=False)

        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            age_embed = self.bert.embeddings.demographic_embeddings(age_ids)
            bmi_embed = self.bert.embeddings.demographic_embeddings(bmi_ids)
            cycle_len_embed = self.bert.embeddings.demographic_embeddings(cycle_len_ids)
            return masked_lm_loss, sequence_output, masked_lm_labels, artcycle_ids.view(-1), output_pred_masks, age_embed, bmi_embed, cycle_len_embed
        else:
            return prediction_scores, artcycle_ids.view(-1)

    def set_output_embeddings_weight(self, new_embeddings_weight):
        self.cls.predictions.decoder.weight = new_embeddings_weight

class EdwardsForMultiLabelPrediction(BertForMaskedLM):
    def __init__(self, config, num_labels):
        super(EdwardsForMultiLabelPrediction, self).__init__(config)

        self.bert = EdwardsModel(config, add_pooling_layer=True)
        # self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, masked_lm_labels=None, artcycle_ids=None, output_pred_masks=None):
    # def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None):
        # _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
        #                              output_all_encoded_layers=False)

        _, pooled_output = self.bert(input_ids, age_ids, bmi_ids, cycle_len_ids, seg_ids, posi_ids, attention_mask,
                                       output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if masked_lm_labels is not None:
            loss_fct = nn.MultiLabelSoftMarginLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), masked_lm_labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits

class EdwardsForMultiLabelPredictionFrozen(BertForMaskedLM):
    def __init__(self, config):
        super(EdwardsForMultiLabelPredictionFrozen, self).__init__(config)

        self.bert = EdwardsModel(config, add_pooling_layer=True)
        # self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, bmi_ids=None, cycle_len_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, masked_lm_labels=None, artcycle_ids=None, output_pred_masks=None):
    # def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None):
        # _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
        #                              output_all_encoded_layers=False)

        _, pooled_output = self.bert(input_ids, age_ids, bmi_ids, cycle_len_ids, seg_ids, posi_ids, attention_mask,
                                       output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        # if masked_lm_labels is not None:
        #     loss_fct = nn.MultiLabelSoftMarginLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), masked_lm_labels.view(-1, self.num_labels))
        #     return loss, logits
        # else:
        #     return logits

        return pooled_output
    

class EdwardsFN(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input, label):
        logits = self.classifier(input)
        loss_fct = nn.MultiLabelSoftMarginLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1, self.num_labels))
        return loss, logits
         

