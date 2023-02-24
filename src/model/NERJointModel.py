# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from model.embed import *
from model.encoders_decoders import *
from model.ModifiedAdaptiveSoftmax import AdaptiveLogSoftmaxWithLoss

from transformers import BertModel, BertPreTrainedModel


class NERJointModel(BertPreTrainedModel):
    def __init__(self, config, params):
        super(NERJointModel, self).__init__(config)
        self.params = params
        self.ner_label_limit = params["ner_label_limit"]
        self.thresholds = params["ner_threshold"]
        self.num_entities = params["mappings"]["nn_mapping"]["num_entities"]        
        self.max_span_width = params["max_span_width"]         
        self.bert = BertModel(config)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sample_size = self.params['sample_size']
        self.stage = 'train'
        
        if self.params['vib']:
            self.params['ib_dim'] = int(config.hidden_size * 3/2)
            self.params['vib_hidden_dim'] = (config.hidden_size*3 + self.params['ib_dim']) // 2
            self.activation = self.params['activation']
            self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}            
            self.ib_dim = self.params['ib_dim']
            self.kl_annealing = self.params['kl_annealing']
            self.hidden_dim = self.params['vib_hidden_dim']           
            intermediate_dim = int((self.hidden_dim+config.hidden_size*3)//2)
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size*3, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation])
            self.beta = self.params['beta']            
            self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim) 
            self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))
            self.entity_classifier = nn.Linear(self.ib_dim, self.num_entities)   
            self.sample_size = self.params['sample_size']
        else:
            self.entity_classifier = nn.Linear(config.hidden_size * 3, self.num_entities)
                
        #since we still need label_ids to evaluate VAE later ...
        self.register_buffer(
                "label_ids",
                torch.tensor(
                    params["mappings"]["nn_mapping"]["mlb"].classes_, dtype=torch.uint8
                ),
            )
       
        # self.apply(self.init_bert_weights)
        self.params = params

        if self.params['ner_vae_joint']:            
            self.w_embed = EmbedLayer(num_embeddings=params['voc_sizes']['word_size'],
                                embedding_dim=params['word_embed_dim'],
                                pretrained=params['pretrained_wordvec'],
                                ignore=params['mappings']['word_map']['<PAD>'],
                                mapping=params['mappings']['word_map'],
                                freeze=params['freeze_words'])
            if self.params['share_mu']:
                self.hid2mu = nn.Linear(config.hidden_size * 3, self.params['latent_size']) 
            else:
                self.hid2mu_span = nn.Linear(config.hidden_size * 3, self.params['latent_size']) 
                self.hid2mu_synonym = nn.Linear(config.hidden_size * 3, self.params['latent_size']) 

            if self.params['share_sigma']:
                self.hid2var = nn.Linear(config.hidden_size * 3, self.params['latent_size'])
            else:
                self.hid2var_span = nn.Linear(config.hidden_size * 3, self.params['latent_size'])
                self.hid2var_synonym = nn.Linear(config.hidden_size * 3, self.params['latent_size'])  
                

            decoder_dim = params['word_embed_dim'] + params['latent_size']

            if self.params['span_reco']:                
                self.latent2hid_span = nn.Linear(self.params['latent_size'], self.params['dec_dim'])
                self.span_reconstruction = LSTMDecoder(in_features=decoder_dim,
                                        h_dec_dim=params['dec_dim'],
                                        layers_num=params['dec_layers'],
                                        dir2=params['dec_bidirectional'],
                                        device=params['device'],
                                        action='sum')
                
            if self.params['syn_gen'] or self.params['span_syn']:                          
                self.latent2hid_synonym = nn.Linear(self.params['latent_size'], self.params['dec_dim'])
                self.syn_generator = LSTMDecoder(in_features=decoder_dim,
                                        h_dec_dim=params['dec_dim'],
                                        layers_num=params['dec_layers'],
                                        dir2=params['dec_bidirectional'],
                                        device=params['device'],
                                        action='sum')

            self.reco_loss = AdaptiveLogSoftmaxWithLoss(params['dec_dim'], params['voc_sizes']['word_size'],
                                                        cutoffs=[round(params['voc_sizes']['word_size']/15),
                                                                 3*round(params['voc_sizes']['word_size']/15)])
    
    def get_span_embeddings(self, embeddings, all_token_masks, max_span_width, device):
        '''
        Enumerate all possible spans and return their corresponding embeddings
        span embeddings = start word emb + end word emb + mean of all word emb
        '''
        flattened_token_masks = all_token_masks.flatten()  # (B * S, )
        flattened_embedding_indices = torch.arange(
            flattened_token_masks.size(0), device=device
        ).masked_select(
            flattened_token_masks
        )  # (all_actual_tokens, )
        # bert
        flattened_embeddings = torch.index_select(
            embeddings.view(-1, embeddings.size(-1)), 0, flattened_embedding_indices
        )  # (all_actual_tokens, H)

        span_starts = (
            torch.arange(flattened_embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, max_span_width)
        )  # (all_actual_tokens, max_span_width)

        flattened_span_starts = (span_starts.flatten())  # (all_actual_tokens * max_span_width, )

        span_ends = span_starts + torch.arange(max_span_width, device=device).view(1, -1)  # (all_actual_tokens, max_span_width)

        flattened_span_ends = (span_ends.flatten())  # (all_actual_tokens * max_span_width, )

        sentence_indices = (
            torch.arange(embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, embeddings.size(1))
        )  # (B, S)

        flattened_sentence_indices = sentence_indices.flatten().masked_select(
            flattened_token_masks
        )  # (all_actual_tokens, )

        span_start_sentence_indices = torch.index_select(
            flattened_sentence_indices, 0, flattened_span_starts
        )  # (all_actual_tokens * max_span_width, )

        span_end_sentence_indices = torch.index_select(
            flattened_sentence_indices,
            0,
            torch.min(
                flattened_span_ends,
                torch.ones(
                    flattened_span_ends.size(),
                    dtype=flattened_span_ends.dtype,
                    device=device,
                )
                * (span_ends.size(0) - 1),
            ),
        )  # (all_actual_tokens * max_span_width, )

        candidate_mask = torch.eq(
            span_start_sentence_indices,
            span_end_sentence_indices,  # Checking both indices is in the same sentence
        ) & torch.lt(
            flattened_span_ends, span_ends.size(0)
        )  # (all_actual_tokens * max_span_width, )

        flattened_span_starts = flattened_span_starts.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        flattened_span_ends = flattened_span_ends.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        span_start_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_starts
        )  # (all_valid_spans, H)

        span_end_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_ends
        )  # (all_valid_spans, H)

        # For computing embedding mean
        mean_indices = flattened_span_starts.view(-1, 1) + torch.arange(
            max_span_width, device=device
        ).view(
            1, -1
        )  # (all_valid_spans, max_span_width)

        mean_indices_criteria = torch.gt(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, max_span_width)
        )  # (all_valid_spans, max_span_width)

        mean_indices = torch.min(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, max_span_width)
        )  # (all_valid_spans, max_span_width)

        span_mean_embeddings = torch.index_select(
            flattened_embeddings, 0, mean_indices.flatten()
        ).view(
            *mean_indices.size(), -1
        )  # (all_valid_spans, max_span_width, H)

        coeffs = torch.ones(
            mean_indices.size(), dtype=embeddings.dtype, device=device
        )  # (all_valid_spans, max_span_width)

        coeffs[mean_indices_criteria] = 0

        span_mean_embeddings = span_mean_embeddings * coeffs.unsqueeze(
            -1
        )  # (all_valid_spans, max_span_width, H)

        span_mean_embeddings = torch.sum(span_mean_embeddings, dim=1) / torch.sum(
            coeffs, dim=-1
        ).view(
            -1, 1
        )  # (all_valid_spans, H)

        combined_embeddings = torch.cat(
                (
                    span_start_embeddings,
                    span_mean_embeddings,
                    span_end_embeddings,
                    # span_width_embeddings,
                ),
                dim=1,
            )  # (all_valid_spans, H * 3 + distance_dim)
        return combined_embeddings
    
    def forward(
            self,            
            all_ids,
            all_token_masks,
            all_attention_masks,
            all_entity_masks,            
            all_span_labels,               
            span_sources=None,
            span_targets=None,
            span_lengths=None,
            span_synonyms=None,
            syn_lengths=None,            
            type='train',
            epoch = 1,
            sampling_type='iid'            
    ):
        device = all_ids.device
        self.stage = type
        # max_span_width = self.max_span_width
  
        #########################
        ## Encoder -- BERT
        #########################

        outputs = self.bert(input_ids=all_ids, attention_mask=all_attention_masks)  # (B, S, H) (B, 128, 768)

        embeddings = outputs.last_hidden_state
        sentence_embedding = outputs.pooler_output
        final_outputs = {}
        # ! REDUCE
        embeddings = self.dropout(embeddings)  # (B, S, H) (B, 128, 768)
        span_embeddings = self.get_span_embeddings(embeddings, all_token_masks, self.max_span_width, device)
        
        all_span_masks = (all_entity_masks > -1) # (B, max_spans)
        gold_masks1 = (all_entity_masks > 1)
        gold_masks2 = all_entity_masks[all_span_masks] > 1        
        vae_z = None
        vae_z_syn = None        
        mu_ = None
        kld = torch.zeros((1,)).to(self.params['device'])
        reco_loss = {'sum': torch.zeros((1,),requires_grad=True).to(self.params['device']),
                         'mean': torch.zeros((1,),requires_grad=True).to(self.params['device'])}
        kld_syn = torch.zeros((1,)).to(self.params['device'])
        reco_syn_loss = {'sum': torch.zeros((1,),requires_grad=True).to(self.params['device']),
                                 'mean': torch.zeros((1,),requires_grad=True).to(self.params['device'])}        
        # mu_ = torch.zeros((self.config.hidden_size, self.params['latent_size'])).to(self.params['device'])          
        if self.params['ner_vae_joint']: 
            if self.params['share_mu']:
                mu_ = self.hid2mu(span_embeddings) #share between span_reco and syn_gen                        
            if self.params['share_sigma']:
                logvar_ = self.hid2var(span_embeddings)
            
            gold_embeddings = span_embeddings[gold_masks2]
            batch_sources = span_sources[gold_masks1] #shape: (all gold entities, max_span_width)
            #########################
            ## Reconstruction of gold entities
            #########################
            if self.params['span_reco']:                                
                if len(batch_sources) > 0:                
                    if self.params['share_mu'] == False:
                        mu_ = self.hid2mu_span(span_embeddings)
                    if self.params['share_sigma'] == False:
                        logvar_ = self.hid2var_span(span_embeddings) 
                    vae_z = self.reparameterisation(mu_, logvar_)                                                            
                    batch_targets = span_targets[gold_masks1] #get the gold span
                    batch_lengths = span_lengths[gold_masks1]
                    self.vae_type = 'span'   
                    vae_z = vae_z[gold_masks2]    
                    mu_gold = mu_[gold_masks2]   #get gold embeddings             
                    kld = self.calc_kld(mu_gold, logvar_[gold_masks2]) 
                    assert not torch.isnan(kld), "The KLD is NAN"   
                    recon_span = self.reconstruction(vae_z, batch_sources, batch_lengths)
                    reco_loss = self.calc_reconstruction_loss(recon_span, batch_targets, batch_lengths)
                    assert not torch.isnan(reco_loss['sum']), "The reco_loss is NAN"       
                           
            ##################################
            ##Synonym generation
            ##Generate synonyms of gold entities
            ##################################            
            if self.params['syn_gen'] or self.params['span_syn']:
                if self.params['share_mu'] == False:
                    mu_ = self.hid2mu_synonym(span_embeddings)
                if self.params['share_sigma'] == False:
                    logvar_ = self.hid2var_synonym(span_embeddings)      
                if type == 'test':                                   
                    vae_z_syn = self.reparameterisation(mu_, logvar_)
                    kld_syn = self.calc_kld(mu_, logvar_) 
                    vae_z_syn = vae_z_syn[gold_masks2]
                else:                                                            
                    if len(batch_sources) > 0:
                        batch_targets = span_synonyms[gold_masks1]
                        batch_lengths = syn_lengths[gold_masks1]                    
                        batch_size = batch_targets.shape            
                        batch_lengths = batch_lengths.view([batch_size[0]*batch_size[1], -1]).squeeze()
                        batch_mask = (batch_lengths > 0) #get rid of padded synonyms
                        count_syns = sum([1 if mask == True else 0 for mask in batch_mask])
                        if count_syns > 0:              
                            self.vae_type = 'syn'  
                            #repeat sources and lengths to match with synonyms, a source has n synonyms --> repeat n
                            #and get rid of padded synonyms using batch_mask
                            mu_gold = mu_[gold_masks2]   #get gold embeddings  
                            mu_syn = mu_gold.unsqueeze(1).repeat(1, batch_size[1], 1)\
                                        .view([batch_size[0]*batch_size[1], -1])[batch_mask]                            
                            if self.params['share_sigma'] == False:
                                input_embeddings = gold_embeddings.unsqueeze(1).repeat(1, batch_size[1], 1)\
                                            .view([batch_size[0]*batch_size[1], -1])[batch_mask]
                                logvar_ = self.hid2var_synonym(input_embeddings)     
                            else:
                                logvar_ = logvar_[gold_masks2]
                                logvar_ = logvar_.unsqueeze(1).repeat(1, batch_size[1], 1)\
                                        .view([batch_size[0]*batch_size[1], -1])[batch_mask]

                            vae_z_syn = self.reparameterisation(mu_syn, logvar_)
                            kld_syn = self.calc_kld(mu_syn, logvar_) 
                            batch_sources = batch_sources.unsqueeze(1).repeat(1, batch_size[1], 1)\
                                        .view([batch_size[0]*batch_size[1], -1]) [batch_mask]           
                            batch_targets = batch_targets.view([batch_size[0]*batch_size[1], -1])[batch_mask]           
                            recon_syn = self.reconstruction(vae_z_syn, batch_sources, batch_lengths[batch_mask])
                            reco_syn_loss = self.calc_reconstruction_loss(recon_syn, batch_targets, batch_lengths[batch_mask])  
                            
        # The number of possible spans is all_valid_spans = K * (2 * N - K + 1) / 2
        # K: max_span_width
        # N: number of tokens
        actual_span_labels = all_span_labels[all_span_masks]  # (all_valid_spans, num_entities)

        all_golds = (actual_span_labels > 0) * self.label_ids

        # Stupid trick
        all_golds, _ = torch.sort(all_golds, dim=-1, descending=True)
        all_golds = torch.narrow(all_golds, 1, 0, self.ner_label_limit)
        all_golds = all_golds.detach().cpu().numpy()
        final_outputs = {'ner_loss': torch.tensor(0), 'e_preds': None,
            'e_golds': all_golds, 'span_masks': all_span_masks, 'bert_emb': embeddings, 
            'sent_emb':sentence_embedding,
            'span_emb': span_embeddings[gold_masks2], 'kld': kld, 'reco_loss':reco_loss,
            'mu_span': mu_, 'latent_z': vae_z, 'priors': None, 'latent_z_syn': vae_z_syn,
            'kld_syn': kld_syn, 
            'gen_loss': reco_syn_loss,
            'true_pred': None         
            }

        if self.params['ner_vae_joint'] and type == 'pretrain': #in the case we pretrain VAEs before NER
            return final_outputs
        ############################
        ##Entity classification 
        ############################
        all_span_masks = (all_entity_masks > -1) # (B, max_spans)

        all_entity_masks = all_entity_masks[all_span_masks] > 0  # (all_valid_spans, )

        # sentence_sections = all_span_masks.sum(dim=-1).cumsum(dim=-1)  # (B, )
       
        # ! REDUCE
        if self.params['ner_reduce']:
            span_embeddings = self.reduce(span_embeddings)
            final_outputs.update({'span_emb': span_embeddings[gold_masks2]})

        if self.params['vib']:
            vib_embeddings = self.mlp(span_embeddings) #only take into account valid spans
            batch_size = vib_embeddings.shape[0]
            mu, std = self.estimate(vib_embeddings, self.emb2mu, self.emb2std)
            mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
            kl_loss = self.kl_div(mu, std, mu_p, std_p)
            z = self.reparameterize(mu, std)
            final_outputs.update({
                    'vib_emb': vib_embeddings[gold_masks2], 
                    'vib_loss': kl_loss,
                    'vib_z': mu[gold_masks2]
                    })
            if self.kl_annealing == "linear":
                beta = min(1.0, epoch*self.beta)                 
            sampled_logits, entity_preds = self.get_logits(z, mu, sampling_type)
            ce_loss = self.sampled_loss(sampled_logits[all_entity_masks], entity_preds[all_entity_masks], actual_span_labels[all_entity_masks].view(-1), sampling_type)
            entity_loss = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * kl_loss  
        else:
            entity_preds = self.entity_classifier(span_embeddings)  # (all_valid_spans, num_entities)
             # Compute entity loss
            entity_loss = F.binary_cross_entropy_with_logits(
                entity_preds[all_entity_masks], actual_span_labels[all_entity_masks]
            )
        
        all_preds = torch.sigmoid(entity_preds)  # (all_valid_spans, num_entities)      
        _, all_preds_top_indices = torch.topk(all_preds, k=self.ner_label_limit, dim=-1)
        # Clear values at invalid positions        
        all_preds[~all_entity_masks, : ] = 0

        # Convert binary value to label ids
        all_preds = (all_preds > self.thresholds) * self.label_ids            
        all_preds = torch.gather(all_preds, dim=1, index=all_preds_top_indices)
        all_preds = all_preds.detach().cpu().numpy()       

        all_aligned_preds = []        
        for _, (preds, golds) in enumerate(zip(all_preds, all_golds)):            
            aligned_preds = []
            pred_set = set(preds) - {0}
            gold_set = set(golds) - {0}
            shared = pred_set & gold_set
            diff = pred_set - shared
            for gold in golds:
                if gold in shared:
                    aligned_preds.append(gold)
                else:
                    aligned_preds.append(diff.pop() if diff else 0)
            all_aligned_preds.append(aligned_preds)

        all_aligned_preds = np.array(all_aligned_preds)

        true_predictions = torch.tensor((all_preds==1)[:,[0]], dtype=torch.bool).squeeze() #take the first prediction only            
        
        final_outputs.update({'ner_loss': entity_loss, 'e_preds':all_aligned_preds,
                    'e_golds': all_golds,'true_pred': true_predictions})
        return final_outputs
    
    ''' The following code is adopted from https://github.com/fenchri/dsre-vae/blob/main/src/bag_net.py
    '''

    def form_decoder_input(self, words):
        """ Word dropout: https://www.aclweb.org/anthology/K16-1002/ """

        random_z2o = torch.rand(words.size()).to(self.params['device'])
        cond1 = torch.lt(random_z2o, self.params['teacher_force'])  # if < word_drop
        cond2 = torch.ne(words, self.params["mappings"]['word_map']['<PAD>']) & \
                torch.ne(words, self.params["mappings"]['word_map']['<SOS>'])  # if != PAD & SOS

        dec_input = torch.where(cond1 & cond2,
                                torch.full_like(words, self.params["mappings"]['word_map']['<UNK>']),
                                words)
        dec_input = self.w_embed(dec_input)
        
        return dec_input
    
    def reconstruction(self, latent_z, decoder_input, decoder_input_length):
        # print ("Input size: ", batch['input'].shape)
        y_vec = self.form_decoder_input(decoder_input)
        # print ("y_vec size: ", y_vec.shape)
        y_vec = torch.cat([y_vec,
                           latent_z.unsqueeze(dim=1).repeat((1, y_vec.size(1), 1))], dim=2)
        if self.vae_type == 'span':
            h_0 = self.latent2hid_span(latent_z).unsqueeze(0)
        else:
            h_0 = self.latent2hid_synonym(latent_z).unsqueeze(0)
        h_0 = h_0.expand(self.params['dec_layers'], h_0.size(1), h_0.size(2))
        c_0 = torch.zeros(self.params['dec_layers'], latent_z.size(0), self.params['dec_dim']).to(self.params['device'])

        if self.vae_type == 'span':
            recon_x, _ = self.span_reconstruction(y_vec, len_=decoder_input_length, hidden_=(h_0, c_0))
        else:
            recon_x, _ = self.syn_generator(y_vec, len_=decoder_input_length, hidden_=(h_0, c_0))
        
        return recon_x
    
    def calc_reconstruction_loss(self, recon_x, decoder_target, decoder_input_length):
        # remove padded
        tmp = torch.arange(recon_x.size(1)).repeat(decoder_input_length.size(0), 1).to(self.params['device'])
        mask = torch.lt(tmp, decoder_input_length[:, None].repeat(1, tmp.size(1)))  # result in (words, dim)

        # Convert to (sentences, words)
        o_vec = self.reco_loss(recon_x[mask], decoder_target[mask])  # (words,)
        o_vec = pad_sequence(torch.split(o_vec.loss, decoder_input_length.tolist(), dim=0),
                             batch_first=True,
                             padding_value=0)
        assert o_vec.size(0) == len(decoder_input_length), "Check output vector of the decoder"
        mean_mean = torch.div(torch.sum(o_vec, dim=1), decoder_input_length.float().to(self.params['device']))
        assert mean_mean.size(0) == len(decoder_input_length), "Check mean vector"
        reco_loss = {'mean': torch.mean(mean_mean),  # mean over words, mean over batch (for perplexity)
                     'sum': torch.mean(torch.sum(o_vec, dim=1))}  # sum over words, mean over batch (sentences)
        return reco_loss

    def reparameterisation(self, mean_, logvar_):
        std = torch.exp(0.5 * logvar_)
        eps = torch.randn_like(std)
        return mean_ + (eps * std)
        

    def inference(self, z, type):
        """
        This inference function uses 'greedy decoding' from Fenia's code
        """
        if type == 'span':
            h0 = self.latent2hid_span(z).unsqueeze(0)
        else:
            h0 = self.latent2hid_synonym(z).unsqueeze(0)
        
        h0 = h0.expand(self.params['dec_layers'], h0.size(1), h0.size(2)).contiguous()
        c0 = torch.zeros(self.params['dec_layers'], z.size(0), self.params['dec_dim']).to(self.params['device']).contiguous()

        # start with start-of-sentence (SOS)
        w_id = torch.empty((1,)).fill_(self.params['mappings']['word_map']['<SOS>']).to(self.params['device']).long()
        gen_sentence = ['<SOS>']

        while (gen_sentence[-1] != '<EOS>') and (len(gen_sentence) <= self.params['max_entity_width']):
            dec_input = self.w_embed(w_id)
            dec_input = torch.cat([dec_input.unsqueeze(0), z.unsqueeze(0)], dim=2)
            if type == 'span':
                next_word_rep, (h0, c0) = self.span_reconstruction(dec_input, hidden_=(h0, c0))
            else:
                next_word_rep, (h0, c0) = self.syn_generator(dec_input, hidden_=(h0, c0))

            logits = self.reco_loss.log_prob(next_word_rep.squeeze(0))
            norm_logits = F.softmax(logits, dim=1)

            w_id = torch.multinomial(norm_logits.squeeze(0), 1)
            # w_id = norm_logits.argmax(dim=1)
            gen_token = self.params['mappings']['rev_word_map'][w_id.item()]
            if gen_token not in ['<UNK>', '<SOS>', '<PAD>']:
                gen_sentence += [gen_token]

        # gen_sentence = ' '.join(gen_sentence[1:-1])
        # print(gen_sentence + '\n')
        return gen_sentence[1:-1]
    
    @staticmethod
    def calc_kld(mu, logvar, mu_prior=None, logvar_prior=None):
        if mu_prior is not None:
            mu_diff = mu_prior.float() - mu
            kld = -0.5 * (1 + logvar - mu_diff.pow(2) - logvar.exp())
        else:
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        kld = torch.sum(kld, dim=1)
        kld = torch.mean(kld)  # sum over dim, mean over batch
        return kld
    

    '''
    The following code was adopted from https://github.com/rabeehk/vibert/blob/master/models.py
    '''

    def get_logits(self, z, mu, sampling_type):
        if sampling_type == "iid":
            logits = self.entity_classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.entity_classifier(mu)
            logits = mean_logits
        return logits, mean_logits

    def sampled_loss(self, logits, mean_logits, labels, sampling_type):        
        if sampling_type == "iid":
            # During the training, computes the loss with the sampled embeddings.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))           
        else:
            # During test time, uses the average value for prediction.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(mean_logits.view(-1), labels.float().view(-1))
           
        return loss

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).cuda()
        return mu + std * z