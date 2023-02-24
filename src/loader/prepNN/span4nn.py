"""Prepare data with span-based for training networks."""

import numpy as np
from collections import namedtuple
import torch
import string

Term = namedtuple('Term', ['id2term', 'term2id', 'id2label', 'text','syns'])


def get_span_index(
        span_start,
        span_end,
        max_span_width,
        max_sentence_length,
        index,
        limit
):
    assert span_start <= span_end
    assert index >= 0 and index < limit
    assert max_span_width > 0
    assert max_sentence_length > 0

    max_span_width = min(max_span_width, max_sentence_length)
    invalid_cases = max(
        0, span_start + max_span_width - max_sentence_length - 1
    )
    span_index = (
            (max_span_width - 1) * span_start
            + span_end
            - invalid_cases * (invalid_cases + 1) // 2
    )
    return span_index * limit + index


def text_decode(sens, tokenizer):
    """Decode text from subword indices using pretrained bert"""

    ids = [id for id in sens]
    orig_text = tokenizer.decode(ids, skip_special_tokens=True)

    return orig_text


def bert_tokenize(sw_sentence, split_line_text, params, tokenizer_encoder):    
    sw_tokens = [token for token, *_ in sw_sentence]    
    num_tokens = len(sw_tokens)

    token_mask = [1] * num_tokens

    shorten = False

    # Account for [CLS] and [SEP] tokens
    if num_tokens > params['max_seq'] - 2:
        num_tokens = params['max_seq'] - 2
        sw_tokens = sw_tokens[:num_tokens]
        token_mask = token_mask[:num_tokens]
        shorten = True

    ids = tokenizer_encoder.convert_tokens_to_ids(["[CLS]"] + sw_tokens + ["[SEP]"])
    token_mask = [0] + token_mask + [0]

    # decode the text if shorten by max sequence
    if shorten:
        orig_text = text_decode(ids, tokenizer_encoder)
    else:
        orig_text = split_line_text

   

    # ! Whether use value 1 for [CLS] and [SEP]
    attention_mask = [1] * len(ids)

    return sw_tokens, ids, num_tokens, token_mask, attention_mask, orig_text


def bert_gpt2_tokenize(split_line_text, tokenizer_decoder):
    # tokenized_text0 = tokenizer_encoder.convert_tokens_to_ids(tokenizer_encoder.tokenize(split_line_text))
    # tokenized_text0 = tokenizer_encoder.add_special_tokens_single_sentence(tokenized_text0)
    # tokenized_text0_length = len(tokenized_text0)
    tokenized_text1 = tokenized_text1_length = None
    if tokenizer_decoder != None:
        gpt2_bos_token = tokenizer_decoder.convert_tokens_to_ids(["<BOS>"])
        gpt2_eos_token = tokenizer_decoder.convert_tokens_to_ids(["<EOS>"])
        tokenized_text1 = tokenizer_decoder.convert_tokens_to_ids(tokenizer_decoder.tokenize(split_line_text))
        tokenized_text1 = tokenizer_decoder.add_special_tokens_single_sentence(tokenized_text1)
        tokenized_text1 = gpt2_bos_token + tokenized_text1 + gpt2_eos_token
        tokenized_text1_length = len(tokenized_text1)

    return tokenized_text1, tokenized_text1_length

def check_valid_span(span_text, params):
    '''
    Check if a span contain punctuations or stop words
    '''
    #if a span starts or ends by a punctuation
    if len(span_text) == 1 and span_text[0] in string.punctuation:
        return False
    
    if span_text[0] in [')', ']', '-', ':', ',', '?', '!', '.'] or \
            span_text[-1] in ['.', ',' ,':','?', '!', '(', '[', '-']:
        return False
    
    #if a span starts or ends by a stop word
    if span_text[0].lower() in params['stop_words'] or span_text[-1] in params['stop_words']:
        return False    
    
    for token in span_text[1:]:
        if token in [',', ':', ';']:
            return False
    return True    

def retrieve_word_id_synonyms(synonyms, params):
    syns_ids = []
    syns_length = []
    for syn in synonyms:
        toks = syn.split(' ')
        toks_id = []
        for tok in toks:
            if tok in params['mappings']['word_map']:
                toks_id.append(params['mappings']['word_map'][tok])
            else:
                toks_id.append(params['mappings']['word_map']['<UNK>'])
        toks_id.append(params['mappings']['word_map']['<EOS>'])
        syns_ids.append(toks_id)
        syns_length.append(len(toks_id))
    return syns_ids, syns_length

def get_batch_data(entities, terms, valid_starts, sw_sentence, words, words_id, sub_to_word, split_line_text,
                   tokenizer_encoder, synSearch, params):
    mlb = params["mappings"]["nn_mapping"]["mlb"]
    # num_labels = params["mappings"]["nn_mapping"]["num_labels"]

    max_entity_width = params["max_entity_width"]    
    max_span_width = params["max_span_width"]

    # bert tokenizer
    _, bert_tokens, num_tokens, token_mask, attention_mask, _ = bert_tokenize(sw_sentence,
                                                                              split_line_text,
                                                                              params,
                                                                              tokenizer_encoder)

    # bert and gpt2 tokenizers
    # TODO: there may be a bug here: if the num_tokens are not matched between the two tokenized outputs (check later)
    # tokenized_text1, tokenized_text1_length = bert_gpt2_tokenize(orig_text, tokenizer_decoder)
    bert_token_length = num_tokens + 2

    # Generate spans here
    span_starts = np.tile(
        np.expand_dims(np.arange(num_tokens), 1), (1, max_span_width)
    )  # (num_tokens, max_span_width)

    span_ends = span_starts + np.expand_dims(
        np.arange(max_span_width), 0
    )  # (num_tokens, max_span_width)

    span_indices = []    
    span_labels = []    
    entity_masks = []    
    span_terms = Term({}, {}, {}, {}, {})

    decoder_span_source = []
    decoder_span_target = []
    decoder_span_length = []   
    generator_syns = []
    generator_lengths = []

    number_of_gold = 0
    number_with_syn = 0
    for span_start, span_end in zip(
            span_starts.flatten(), span_ends.flatten()
    ):
        if span_start >= 0 and span_end < num_tokens:
            span_label = []  # No label
            span_term = []            
            entity_mask = 1          
            if span_end - span_start + 1 > max_entity_width:
                entity_mask = 0

            real_start = sub_to_word[span_start]
            real_end = sub_to_word[span_end]
            span_text = split_line_text.split(' ')[real_start:real_end+1]

            # Ignore spans containing incomplete words
            valid_span = True
            if not params['predict']:
                if span_start not in valid_starts or (span_end + 1) not in valid_starts:
                    # Ensure that there is no entity label here              
                    assert (span_start, span_end) not in entities
                    entity_mask = 0                                   
                    valid_span = False
            
            synonyms = []
            temp = ' '.join(span_text).lower()          
            if valid_span:
                if (span_start, span_end) in entities:
                    span_term = terms[(span_start, span_end)]
                    span_label = entities[(span_start, span_end)]
                    number_of_gold += 1                    
                    entity_mask = 2                                  
                if synSearch!=None:                                                          
                    if temp in synSearch:
                        synonyms = synSearch[temp]
                        if len(synonyms) > 0: 
                            number_with_syn += 1                    
                    if params['span_syn']: #consider a span as its synonym
                        synonyms.append(temp)        
            
            #create spans to input for the decoder            
            span_org = words_id[real_start:real_end+1]
            span_source = [params['mappings']['word_map']['<SOS>']] + span_org
            span_target = span_org + [params['mappings']['word_map']['<EOS>']] 
            span_length = len(span_source) 
            span_org = ' '.join(words[real_start:real_end+1])
            
            #get word ids for the synonyms 
            #record the synonym length for the decoder
            synonyms_id, synonyms_length = retrieve_word_id_synonyms(synonyms, params)
            generator_syns.append(synonyms_id)
            generator_lengths.append(synonyms_length)
                 
            # assert len(span_label) <= params["ner_label_limit"], "Found an entity having a lot of types"
            if len(span_label) > params["ner_label_limit"]:
                print('over limit span_label', span_term)

            # For multiple labels
            for idx, (_, term_id) in enumerate(
                    sorted(zip(span_label, span_term), reverse=True)[:params["ner_label_limit"]]):
                span_index = get_span_index(span_start, span_end, max_span_width, num_tokens, idx,
                                            params["ner_label_limit"])

                span_terms.id2term[span_index] = term_id
                span_terms.term2id[term_id] = span_index                                        
                # add entity type
                term_label = params['mappings']['nn_mapping']['id_tag_mapping'][span_label[0]]
                span_terms.id2label[span_index] = term_label
                #to use at the end to evaluate bleu score
                span_terms.text[term_id] = span_text
                span_terms.syns[term_id] = synonyms

            span_label = mlb.transform([span_label])[-1]

            span_indices += [(span_start, span_end)] * params["ner_label_limit"]
            span_labels.append(span_label)            
            entity_masks.append(entity_mask)
            decoder_span_source.append(span_source)
            decoder_span_target.append(span_target)
            decoder_span_length.append(span_length)
    
    return {
        'bert_token': bert_tokens,
        'token_mask': token_mask,
        'attention_mask': attention_mask,
        'span_indices': span_indices,
        'span_labels': span_labels,
        'entity_masks': entity_masks,        
        'span_terms': span_terms,
        'bert_token_length': bert_token_length,
        'span_source': decoder_span_source,
        'span_target': decoder_span_target,
        'span_length': decoder_span_length,
        'span_synonyms': generator_syns,
        'span_syn_lengths': generator_lengths,
    }, number_of_gold, number_with_syn


def get_nn_data(entitiess, termss, valid_startss, sw_sentences, wordss, word_idss, sub_to_words,
                split_line_text_, tokenizer_encoder, synSearch, params):
    samples = []

    # filter by sentence length
    dropped = 0
    total_gold = total_syn = 0
    # for idx, sw_sentence in enumerate(sw_sentences):
    span_priors = None
    for idx, split_line_text in enumerate(split_line_text_):
        
        if len(split_line_text) < 1:
            dropped += 1
            continue

        if len(split_line_text.split()) > params['block_size']:
            dropped += 1
            continue

        sw_sentence = sw_sentences[idx]
        sub_to_word = sub_to_words[idx]
        words_id = word_idss[idx]
        words = wordss[idx]

        entities = entitiess[idx]
        terms = termss[idx]
        valid_starts = valid_startss[idx]

        sample, num_gold, num_syn = get_batch_data(entities, terms, valid_starts, sw_sentence, words, words_id, sub_to_word,
                                split_line_text,
                                tokenizer_encoder,    
                                synSearch,                           
                                params)        
        total_gold += num_gold
        total_syn += num_syn
        samples.append(sample)

    print('max_seq', params['max_seq'])

    bert_tokens = [sample["bert_token"] for sample in samples]
    all_token_masks = [sample["token_mask"] for sample in samples]
    all_attention_masks = [sample["attention_mask"] for sample in samples]
    all_span_indices = [sample["span_indices"] for sample in samples]
    all_span_labels = [sample["span_labels"] for sample in samples]    
    all_entity_masks = [sample["entity_masks"] for sample in samples]
    all_span_terms = [sample["span_terms"] for sample in samples]    
    bert_token_lengths = [sample["bert_token_length"] for sample in samples]    
    span_sources = [sample["span_source"] for sample in samples]
    span_targets = [sample["span_target"] for sample in samples]
    span_lengths = [sample["span_length"] for sample in samples]
    span_synonyms =  [sample["span_synonyms"] for sample in samples]
    span_syn_lengths =  [sample["span_syn_lengths"] for sample in samples]
    
    print("dropped sentences: ", dropped)
    print ("Total gold: ", total_gold)
    print("Total syn: ", total_syn)

    return {        
        'bert_tokens': bert_tokens,
        'token_mask': all_token_masks,
        'attention_mask': all_attention_masks,
        'span_indices': all_span_indices,
        'span_labels': all_span_labels,
        'entity_masks': all_entity_masks,
        'span_terms': all_span_terms,
        'bert_token_lengths': bert_token_lengths,
        'span_sources': span_sources,
        'span_targets': span_targets,
        'span_lengths': span_lengths,
        'span_priors': span_priors,
        'span_synonyms': span_synonyms,
        'span_syn_lengths': span_syn_lengths,        
    }
