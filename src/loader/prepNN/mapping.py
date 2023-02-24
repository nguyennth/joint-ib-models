"""Generate mappings"""

import itertools
from collections import OrderedDict
from collections import Counter
import numpy as np

from loader.prepData.entity import entity_tags


def _generate_mapping(list_of_elems):
    """
        :param list_of_elems: list of elements (single or nested)
        :returns
            dictionary with a unique id for each element
    """
    # list of lists
    elem_count = OrderedDict()
    if all(isinstance(el, list) for el in list_of_elems):
        for item in itertools.chain.from_iterable(list_of_elems):
            if item not in elem_count:
                elem_count[item] = 1
            else:
                elem_count[item] += 1
    # single lists
    else:
        for item in list_of_elems:
            if item not in elem_count:
                elem_count[item] = 1
            else:
                elem_count[item] += 1
    elem_count = sorted(elem_count.items(), key=lambda x: x[1])  # sort from low to high freq
    mapping = OrderedDict([(elem, i) for i, (elem, val) in enumerate(elem_count)])
    rev_mapping = OrderedDict([(v, k) for k, v in mapping.items()])
    return mapping, rev_mapping, len(elem_count)


def _find_singletons(list_of_elems, args, min_w_freq):
    """
        :param list_of_elems: list of all words in a train dataset
        :returns
            number of words with frequency = 1
    """
    elem_count = Counter([x for x in list_of_elems])
    unique_args = list(set(itertools.chain.from_iterable([a.split(' ') for a in args])))
    singles = [elem for elem, val in elem_count.items() if ((val <= min_w_freq) and (elem not in unique_args))]
    return singles


def generate_map(data_struct, data_struct_dev, data_struct_test, params):  # add test for mlee

    # 1. words mapping
    #Nhung added for the decoder
    words = data_struct['sentences']['sent_words']
    words = [['<PAD>', '<SOS>', '<EOS>', '<UNK>']] + words
    words_train = data_struct['sentences']['words']
    word_map, rev_word_map, word_size = _generate_mapping(words)

    # 2. ..
    # labels of entity (in .a1)
    argumentsT = data_struct['entities']['arguments']

    # labels of trigger (in .a2)
    argumentsTR = data_struct['triggers']['arguments']
    arguments = argumentsT + argumentsTR
    singlesW = _find_singletons(words_train, arguments, params['min_w_freq'])

    typesTR = data_struct['terms']['typesTR']
    typesTR.extend(data_struct_dev['terms']['typesTR'])

    typesT = data_struct['terms']['typesT']
    typesT.extend(data_struct_dev['terms']['typesT'])

    # add for test: fig bug for mlee
    typesTR.extend(data_struct_test['terms']['typesTR'])
    typesT.extend(data_struct_test['terms']['typesT'])

    all_types = []
    for type in typesTR:
        if type not in all_types:
            all_types.append(type)

    for type in typesT:
        if type not in all_types:
            all_types.append(type)

    type_map = {type: id for id, type in enumerate(all_types)}
    rev_type_map = {id: type for type, id in type_map.items()}
    type_size = len(type_map)

    typeTR_map = {}
    for type, id in type_map.items():
        if type in typesTR:
            typeTR_map[type] = id
    rev_typeTR_map = {id: type for type, id in typeTR_map.items()}
    # typeTR_size = len(typeTR_map)

    rev_tag_map, tag_map, _, _ = entity_tags(rev_type_map)

    tag_size = len(tag_map)

    trTypeIds = [id for id in rev_typeTR_map]

    tagsTR = data_struct['terms']['tagsTR']
    tagsTR2 = data_struct_dev['terms']['tagsTR']
    tagsTR.extend([tag for tag in tagsTR2 if tag not in tagsTR])
    rev_tag_mapTR = {tag_map[tag]: tag for tag in tagsTR}

    tag_mapTR = {tag: id for id, tag in rev_tag_mapTR.items()}
    trTagsIds = [tag for tag in rev_tag_mapTR]

    tag2type = data_struct['terms']['tags2types']
    tag2type2 = data_struct_dev['terms']['tags2types']
    for tag in tag2type2:
        if tag not in tag2type:
            tag2type[tag] = tag2type2[tag]
    tag2type_map = OrderedDict()
    for tag in tag2type:
        if tag != 'O':
            type = tag2type[tag]
            tag2type_map[tag_map[tag]] = type_map[type]
    tag2type_map[0] = -1  # tag O

    tag2type = np.zeros(tag_size, np.int32)
    for tag, type in tag2type_map.items():
        tag2type[tag] = type

    # 3. pos map
    all_sents = data_struct['sentences']['sentences']
    all_sents.extend(data_struct_dev['sentences']['sentences'])

    length = [len([w for w in s.split()]) for s in all_sents]
    ranges = [list(map(str, list(range(-l + 1, l)))) for l in length]
    if params['include_nested']:
        ranges.append(['inner'])  # encode nestedness embeddings
        ranges.append(['outer'])
    pos_map, rev_pos_map, pos_size = _generate_mapping(ranges)

    # return
    params['voc_sizes'] = {'word_size': word_size,
                           'etype_size': type_size,
                           'tag_size': tag_size,
                           'pos_size': pos_size
                           }
    params['mappings'] = {'word_map': word_map, 'rev_word_map': rev_word_map,
                          'type_map': type_map, 'rev_type_map': rev_type_map,
                          'typeTR_map': typeTR_map, 'rev_typeTR_map': rev_typeTR_map,
                          'tag_map': tag_map, 'rev_tag_map': rev_tag_map,
                          'tag_mapTR': tag_mapTR, 'rev_tag_mapTR': rev_tag_mapTR,
                          'tag2type_map': tag2type,
                          'pos_map': pos_map, 'rev_pos_map': rev_pos_map
                          }
    params['trTags_Ids'] = trTagsIds
    params['trTypes_Ids'] = trTypeIds
    params['words_train'] = words_train
    params['singletons'] = singlesW
    params['max_sent_len'] = np.maximum(data_struct['sentences']['max_sent_len'],
                                        data_struct_dev['sentences']['max_sent_len'])

    return params



def _elem2idx(list_of_elems, map_func):
    """
        :param list_of_elems: list of lists
        :param map_func: mapping dictionary
        :returns
            list with indexed elements
    """
    # fix bug for mlee
    # return [[map_func[x] if x in map_func else map_func["O"] for x in list_of] for list_of in list_of_elems]
    return [[map_func[x] for x in list_of] for list_of in list_of_elems]
