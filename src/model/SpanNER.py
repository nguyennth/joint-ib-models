from collections import namedtuple
import copy

import numpy as np
import torch
from torch import nn

from model.NERJointModel import NERJointModel

cpu_device = torch.device("cpu")
Term = namedtuple('Term', ['id2term', 'term2id', 'id2label'])

class SpanNER(nn.Module):    
    def __init__(self, params):
        super(SpanNER, self).__init__()                        
        self.device = params['device']
        self.params = params
        self.entity_id = 1
        self.NER_layer = NERJointModel.from_pretrained(params['bert_model'],params=params)
    
    def _calculate_loss_syn(self, ner_loss, reco_loss, reco_kld, gen_loss, gen_kld, n_epoch, kl_w):
        """To calculate the total loss from the layers' loss"""
        # losses_ = []    
        results_loss = {'ner_loss': 0, 'reco_loss':0, 'kld_reco': 0, 'gen_loss': 0, 'kld_gen': 0}        
        results_loss['ner_loss'] =  ner_loss.item()
        reco_loss_org = reco_loss['sum'] + kl_w * reco_kld
        gen_loss_org = gen_loss['sum'] + kl_w * gen_kld        
        total_loss = 0
        if self.params['span_reco']: #vae_loss is calculated with batch_size already!                        
            results_loss['reco_loss'] = reco_loss_org.item()
            results_loss['kld_reco'] =  (kl_w * reco_kld).item()            
        
        if self.params['syn_gen'] or self.params['span_syn']:                        
            results_loss['gen_loss'] = gen_loss_org.item()
            results_loss['kld_gen'] = (kl_w * gen_kld).item()

        if self.params['ner_vae_joint']:            
            if reco_loss_org != 0 and gen_loss_org !=0:
                total_loss = reco_loss_org + gen_loss_org
            elif reco_loss_org != 0:
                total_loss = reco_loss_org 
            else:
                total_loss = gen_loss_org   

            # total_loss = (1-self.params['task_weight']) * total_loss      
            total_loss = self.params['gama'] * total_loss      
            
            if n_epoch >= self.params['ner_start_epoch']:                                
                total_loss = ner_loss + total_loss   
                    
        else:
            total_loss = ner_loss   
        
        return total_loss, results_loss
    

    def forward(self, batch_input, epoch=0, kl_w=0, stage='train', sampling_type='iid'):
        self.training_type = stage
        nn_bert_tokens, nn_token_mask, nn_attention_mask, nn_span_indices, \
                nn_span_labels, nn_entity_masks, span_terms, \
                span_sources, span_targets, span_lengths, max_span_labels, \
                span_synonyms, span_syn_lengths = batch_input

        outputs = self.NER_layer(                    
                all_ids=nn_bert_tokens,
                all_token_masks=nn_token_mask,
                all_attention_masks=nn_attention_mask,
                all_entity_masks=nn_entity_masks,                
                all_span_labels=nn_span_labels,        
                span_sources=span_sources,
                span_targets=span_targets,
                span_lengths=span_lengths,
                span_synonyms=span_synonyms,
                syn_lengths=span_syn_lengths,                   
                type=stage,
                epoch=epoch+1,
                sampling_type=sampling_type
        )
        
        total_loss, detailed_loss = self._calculate_loss_syn(outputs['ner_loss'], 
                        outputs['reco_loss'], outputs['kld'], outputs['gen_loss'], 
                        outputs['kld_syn'],epoch, kl_w)

        e_preds = outputs['e_preds']
        e_golds = outputs['e_golds']                                                                
        span_masks = outputs['span_masks']           
        # entity output
        ner_preds = {}
        sentence_sections = span_masks.sum(dim=-1).cumsum(dim=-1)
        # #posterior z of gold spans
        labels_z = []        
        if self.params['predict']:                        
            for id, labels in enumerate(e_golds):
                for label_id in labels:
                    if label_id >= 1: 
                        labels_z.append(self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id])

        # ner_preds['terms'] = span_terms
        ner_preds['gold_terms'] = copy.deepcopy(span_terms)
        ner_preds['span_indices'] = nn_span_indices         
        ner_preds['latent_z'] = outputs['latent_z']
        ner_preds['labels_z'] = labels_z            
        ner_preds['vib_z'] = outputs['vib_z'] if 'vib_z' in outputs else None
        ner_preds['vib_emb'] = outputs['vib_emb'] if 'vib_emb' in outputs else None            
        ner_preds['latent_z_syn'] = outputs['latent_z_syn']
        ner_preds['gold_embs'] = outputs['span_emb']

        if self.training_type == 'pretrain' or self.training_type == 'test_syn':                        
            return total_loss, detailed_loss, ner_preds
        
        # Pred of each span
        e_preds = np.split(e_preds.astype(int), sentence_sections)
        e_preds = [pred.flatten() for pred in e_preds]
        ner_preds['preds'] = e_preds
        entity_idx = self.entity_id
        span_terms = []        
        for span_preds in e_preds:
            doc_spans = Term({},{},{})            
            for pred_idx, label_id in enumerate(span_preds):
                if label_id > 0:                    
                    term = "T" + str(entity_idx)
                    doc_spans.id2term[pred_idx] = term
                    doc_spans.term2id[term] = pred_idx
                    entity_idx += 1
            span_terms.append(doc_spans)

        self.entity_id = entity_idx
        ner_preds['terms'] = span_terms

        return total_loss, detailed_loss, ner_preds
    
    