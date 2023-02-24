"""Prepare data for training networks."""

import collections
from collections import OrderedDict
import operator

from sklearn.preprocessing import MultiLabelBinarizer

from loader.prepNN.sent2net import prep_sentences
from loader.prepNN.ent2net import entity2network
from loader.prepNN.mapping import _elem2idx
from loader.prepNN.span4nn import get_nn_data


def data2network(data_struct, data_type, tokenizer_encoder, params):
    # input
    sent_words = data_struct['sentences']

    # words
    org_sent_words = sent_words['sent_words']
    sent_words = prep_sentences(sent_words, data_type, params)
    wordsIDs = _elem2idx(sent_words, params['mappings']['word_map'])

    all_sentences = []

    events_map = collections.defaultdict()

    for xx, sid in enumerate(data_struct['input']):

        # input
        sentence_data = data_struct['input'][sid]

        # document id
        fid = sid.split(':')[0]

        # words to ids
        # words = sentence_data['words']
        word_ids = wordsIDs[xx]
        words = org_sent_words[xx]

        # split line text
        split_line_text = data_struct['input'][sid]['sentence']

        # entity
        readable_e, idxs, ents, toks2, etypes2ids, entities, \
            sw_sentence, sub_to_word, subwords, valid_starts, \
            tagsIDs, tagsTR, terms = entity2network(sentence_data, words, params, tokenizer_encoder)

        # return
        sentence_vector = OrderedDict()
        sentence_vector['fid'] = fid
        sentence_vector['ents'] = ents
        sentence_vector['word_ids'] = word_ids
        sentence_vector['words'] = words
        sentence_vector['offsets'] = sentence_data['offsets']
        sentence_vector['e_ids'] = idxs
        sentence_vector['tags'] = tagsIDs
        sentence_vector['tagsTR'] = tagsTR
        sentence_vector['etypes2'] = etypes2ids
        sentence_vector['toks2'] = toks2
        sentence_vector['raw_words'] = sentence_data['words']

        # nner
        sentence_vector['entities'] = entities
        sentence_vector['sw_sentence'] = sw_sentence
        sentence_vector['terms'] = terms        
        sentence_vector['sub_to_word'] = sub_to_word
        sentence_vector['subwords'] = subwords
        sentence_vector['split_line_text'] = split_line_text
        sentence_vector['valid_starts'] = valid_starts

        all_sentences.append(sentence_vector)

    return all_sentences, events_map


def sort_len(all_sentences):
    # print('SORT SENTENCE LENGTH')

    corpus = [(t, len(t['subwords'])) for t in all_sentences]
    corpus.sort(key=operator.itemgetter(1), reverse=True)
    texts = [x for x, _ in corpus][:2000000]

    # print('DONE SORT')

    return texts


def filter_len(all_sentences, max_len):
    # print('SORT SENTENCE LENGTH')

    corpus = [(t, len(t['subwords'])) for t in all_sentences if len(t['subwords']) <= max_len]
    corpus.sort(key=operator.itemgetter(1))
    texts = [x for x, _ in corpus][:2000000]

    # print('DONE SORT')

    return texts


def torch_data_2_network(cdata2network, tokenizer_encoder, events_map, params, do_get_nn_data, synSearch):
    """ Convert object-type data to torch.tensor type data, aim to use with Pytorch
    """
    etypes = [data['etypes2'] for data in cdata2network]

    # nner
    entitiess = [data['entities'] for data in cdata2network]
    termss = [data['terms'] for data in cdata2network]
    valid_startss = [data['valid_starts'] for data in cdata2network]
    fids = [data['fid'] for data in cdata2network]
    wordss = [data['words'] for data in cdata2network]
    word_ids = [data['word_ids'] for data in cdata2network]
    offsetss = [data['offsets'] for data in cdata2network]
    sub_to_words = [data['sub_to_word'] for data in cdata2network]
    subwords = [data['subwords'] for data in cdata2network]

    # list of subwords for each sentence
    sw_sentences = [data['sw_sentence'] for data in cdata2network]

    # list of sentences
    split_line_text_ = [data['split_line_text'] for data in cdata2network]

    # User-defined data
    if not params["predict"]:
        id_tag_mapping = params["mappings"]["nn_mapping"]["id_tag_mapping"]
        trigger_ids = params["mappings"]["nn_mapping"]["trTypes_Ids"]

        mlb = MultiLabelBinarizer()
        mlb.fit([sorted(id_tag_mapping)[1:]])  # [1:] skip label O

        params["mappings"]["nn_mapping"]["mlb"] = mlb
        params["mappings"]["nn_mapping"]["num_labels"] = len(mlb.classes_)

        params["max_span_width"] = params["max_entity_width"]

        params["mappings"]["nn_mapping"]["full_labels"] = sorted([v for k, v in id_tag_mapping.items() if k > 0])
        params["mappings"]["nn_mapping"]["trigger_labels"] = sorted(
            [v for k, v in id_tag_mapping.items() if k in trigger_ids])

        params["mappings"]["nn_mapping"]["num_triggers"] = len(params["mappings"]["nn_mapping"]["trigger_labels"])
        params["mappings"]["nn_mapping"]["num_entities"] = params["mappings"]["nn_mapping"]["num_labels"] - \
                                                           params["mappings"]["nn_mapping"]["num_triggers"]

    if do_get_nn_data:        
        nn_data = get_nn_data(entitiess, termss, valid_startss, sw_sentences, wordss, word_ids, sub_to_words,
                              split_line_text_, tokenizer_encoder, 
                              synSearch, params)

        return {'nn_data': nn_data, 'etypes': etypes, 'fids': fids, 'words': wordss, 'offsets': offsetss,
                'sub_to_words': sub_to_words, 'subwords': subwords, 'entities': entitiess}
    else:
        return {'termss': termss, 'sw_sentences': sw_sentences,
                'tokenizer': tokenizer_encoder, 'events_map': events_map, 'params': params, 'etypes': etypes, 'fids': fids,
                'words': wordss, 'offsets': offsetss, 'sub_to_words': sub_to_words, 'subwords': subwords,
                'entities': entitiess}
