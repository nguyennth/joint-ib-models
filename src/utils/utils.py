import argparse
import copy
import json
import logging
import os
import pickle
import pprint
import random
import re
import shutil
from collections import OrderedDict
from datetime import datetime
from glob import glob
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import yaml
from time import time

from transformers import (AutoTokenizer, BertTokenizer, GPT2Tokenizer, RobertaTokenizer,T5Tokenizer)

MODEL_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
    't5': T5Tokenizer,
    'gpt2':GPT2Tokenizer,
}

logger = logging.getLogger(__name__)

def _truncate(arr, max_length):
    while True:
        total_length = len(arr)
        if total_length <= max_length:
            break
        else:
            arr.pop()


def _padding(arr, max_length, padding_idx=-1):
    while len(arr) < max_length:
        arr.append(padding_idx)


def _to_tensor(arr, params):
    return torch.tensor(arr, device=params['device'])


def _to_torch_data(arr, max_length, params, padding_idx=-1):
    for e in arr:
        _truncate(e, max_length)
        _padding(e, max_length, padding_idx=padding_idx)
    return _to_tensor(arr, params)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


def path(*paths):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), *paths))


def make_dirs(*paths):
    os.makedirs(path(*paths), exist_ok=True)


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def deserialize(filename):
    with open(path(filename), "rb") as f:
        return pickle.load(f)

def focal_loss (prediction, target, alpha, gamma):
    #implementation from this https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327
    BCE_loss = F.binary_cross_entropy_with_logits(prediction, target)
    pt = torch.exp(-BCE_loss) # prevents nans when probability 0
    focal_loss = alpha * (1-pt)** gamma * BCE_loss
    return focal_loss.mean()

def serialize(obj, filename):
    make_dirs(os.path.dirname(filename))
    with open(path(filename), "wb") as f:
        pickle.dump(obj, f)


def _parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='experiments/ncbi/ncbi-train.yaml', help='yaml file')
    parser.add_argument('--x0', type=int, default=8000, help='The middle peak to anneal function')
    parser.add_argument('--gpu', type=int, default=0, help="GPU")
    parser.add_argument('--start_epoch', type=int, default=0, help="Start epoch, if start_epoch >0, resume from a pre-trained epoch")
    parser.add_argument('--epoch', type=int, default=5, help="Number of epoch")
    parser.add_argument('--ner_start_epoch', type=int, default=3, help="Number of epoch to pretrain VAE before NER")    
    parser.add_argument('--syn_gen', default=True, action='store_true')
    parser.add_argument('--span_syn', default=False, action='store_true')
    parser.add_argument('--span_reco', default=True, action='store_true')
    parser.add_argument('--vib', default=True, action='store_true')
    parser.add_argument('--share_mu', default=True, action='store_true')
    parser.add_argument('--share_sigma', default=False, action='store_true')    
    args = parser.parse_args()
    return vars(args)

def _ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
        Load parameters from yaml in order
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    # print(dict(yaml.load(stream, OrderedLoader).items()))

    return yaml.load(stream, OrderedLoader)


def _print_config(config, config_path):
    """Print config in dictionary format"""
    print("\n====================================================================\n")
    print('RUNNING CONFIG: ', config_path)
    print('TIME: ', datetime.now())

    for key, value in config.items():
        print(key, value)

    return


def dicard_invalid_nes(terms, sentences):
    """
    Discard incomplete tokenized entities.
    """
    text = ' '.join(sentences)
    valid_terms = []
    count = 0
    for term in terms:
        start, end = int(term[2]), int(term[3])
        if start == 0:
            if text[end] == ' ':
                valid_terms.append(term)
            else:
                count += 1
            #    print('Context:{}\t{}'.format(text[start:end + 1], term))
        elif text[start - 1] == ' ' and text[end] == ' ':
            valid_terms.append(term)
        else:
            count += 1
        #    print('Context:{}\t{}'.format(text[start-1:end+1], term))
    return valid_terms, count


def _humanized_time(second):
    """
        Returns a human readable time.
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def is_best_epoch(prf_):
    fs = []
    for epoch, (p, r, f) in enumerate(prf_):
        fs.append(f)

    if len(fs) == 1:
        return True

    elif max(fs[:-1]) < fs[-1]:
        return True

    else:
        return False


def extract_scores(task, prf_):
    ps = []
    rs = []
    fs = []
    for epoch, (p, r, f) in enumerate(prf_):
        ps.append(p)
        rs.append(r)
        fs.append(f)

    maxp = max(ps)
    maxr = max(rs)
    maxf = max(fs)

    maxp_index = ps.index(maxp) + 1
    maxr_index = rs.index(maxr) + 1
    maxf_index = fs.index(maxf) + 1

    print('TASK: ', task)
    print('precision: ', ps)
    print('recall:    ', rs)
    print('fscore:    ', fs)
    print('best precision/recall/fscore [epoch]: ', maxp, ' [', maxp_index, ']', '\t', maxr, ' [', maxr_index, ']',
          '\t', maxf, ' [', maxf_index, ']')
    # print()

    return (maxp, maxr, maxf)


def write_best_epoch(result_dir):
    # best_dir = params['ev_setting'] + params['ev_eval_best']
    best_dir = result_dir + 'ev-best/'

    if os.path.exists(best_dir):
        os.system('rm -rf ' + best_dir)
    # else:
    #     os.makedirs(best_dir)

    current_dir = result_dir + 'ev-last/'

    shutil.copytree(current_dir, best_dir)


def dumps(obj):
    if isinstance(obj, dict):
        return json.dumps(obj, indent=4, ensure_ascii=False)
    elif isinstance(obj, list):
        return pprint.pformat(obj)
    return obj


def debug(*args, **kwargs):
    print(*map(dumps, args), **kwargs)


def get_max_entity_id(span_terms):
    max_id = 0
    for items in span_terms:
        for item in items.term2id:
            matcher = re.search(r"^T(?!R)\S*?(\d+)(?=\s)", item)
            if matcher:
                max_id = max(max_id, int(matcher.group(1)))
    return max_id


def gen_nn_mapping(tag2id_mapping, tag2type_map, trTypes_Ids):
    nn_tr_types_ids = []
    nn_tag_2_type = {}
    tag_names = []
    for tag, _id in tag2id_mapping.items():
        if tag.startswith("I-"):
            continue
        tag_names.append(re.sub("^B-", "", tag))
        if tag2type_map[_id] in trTypes_Ids:
            nn_tr_types_ids.append(len(tag_names) - 1)

        nn_tag_2_type[len(tag_names) - 1] = tag2type_map[_id]

    id_tag_mapping = {k: v for k, v in enumerate(tag_names)}
    tag_id_mapping = {v: k for k, v in id_tag_mapping.items()}

    # For multi-label nner
    assert all(_id == tr_id for _id, tr_id in
               zip(sorted(id_tag_mapping)[1:], nn_tr_types_ids)), "Trigger IDS must be continuous and on the left side"
    return {'id_tag_mapping': id_tag_mapping, 'tag_id_mapping': tag_id_mapping, 'trTypes_Ids': nn_tr_types_ids,
            'tag2type_map': nn_tag_2_type}


def padding_samples(ids_, token_mask_, attention_mask_, span_indices_, span_labels_, 
            span_sources, span_targets, span_lengths, 
            entity_masks_, span_synonyms, span_syn_lengths,         
            params):
    # count max lengths:
    max_seq = 0
    for ids in ids_:
        max_seq = max(max_seq, len(ids))

    max_span_labels = 0
    number_of_syns = 10
    number_of_syn_tokens = params['max_span_width']
    
    for idx, span_labels in enumerate(span_labels_):
        max_span_labels = max(max_span_labels, len(span_labels))
        # number_of_syns = max(number_of_syns, max([len(span_syn) for span_syn in span_synonyms[idx]]))
        # number_of_syn_tokens = max(number_of_syn_tokens, max ([max(sl) if len(sl) > 0 else 0 for sl in span_syn_lengths[idx]]))

    for idx, (
            ids, token_mask, attention_mask, span_indices, span_labels, span_source, span_target, span_length,
            span_syns, span_syn_lens, entity_masks) in enumerate(
        zip(
            ids_,
            token_mask_,
            attention_mask_,
            span_indices_,
            span_labels_,
            span_sources,
            span_targets,
            span_lengths,            
            span_synonyms,
            span_syn_lengths,
            entity_masks_,
            )):
        padding_size = max_seq - len(ids)

        # Zero-pad up to the sequence length
        ids += [0] * padding_size
        token_mask += [0] * padding_size
        attention_mask += [0] * padding_size


        # Padding for span indices and labels
        num_padding_spans = max_span_labels - len(span_labels)

       
        span_indices += [(-1, -1)] * (num_padding_spans * params["ner_label_limit"])
        span_labels += [np.zeros(params["mappings"]["nn_mapping"]["num_labels"])] * num_padding_spans 

        # span_labels_match_rel += [-1] * num_padding_spans
        entity_masks += [-1] * num_padding_spans
        # trigger_masks += [-1] * num_padding_spans


        assert len(ids) == max_seq
        assert len(token_mask) == max_seq
        assert len(attention_mask) == max_seq
        assert len(span_indices) == max_span_labels * params["ner_label_limit"]
        assert len(span_labels) == max_span_labels
        assert len(entity_masks) == max_span_labels     

        assert len(span_source) == len (span_target)
        assert len(span_target) == len(span_length)
        #padding span_sources, targets and length for the decoder
        #need to padding two things: the number of spans in each sentence, and the length of each spans
        #pad the length of each span              
        
        for idx1, (span_s, span_t, span_l, span_syn, span_syn_l) \
                in enumerate (zip(span_source, span_target, span_length, span_syns, span_syn_lens)):                             
            if len(span_s) > params["max_span_width"] + 1:            
                # print ("span is longer than max_span_width", span_s )
                span_s = span_s[:params["max_span_width"] + 1]
                span_t = span_t[:params["max_span_width"]] + [params['mappings']['word_map']['<EOS>']]
                span_l = params["max_span_width"] + 1                
            else:
                num_pad_tok = params["max_span_width"] + 1 - len(span_s) # +1 because of the <SOS> or <EOS> in each span
                span_s += [params['mappings']['word_map']['<PAD>']]*num_pad_tok
                span_t += [params['mappings']['word_map']['<PAD>']]*num_pad_tok
            
            # pad the number of token in each synonym
            for idx2, syn in enumerate (span_syn):
                if len(syn) > number_of_syn_tokens + 1:
                    syn = syn[:number_of_syn_tokens] + [params['mappings']['word_map']['<EOS>']]
                    span_syn_l[idx2] = number_of_syn_tokens + 1
                else:
                    pad_syn = number_of_syn_tokens + 1 - len(syn) # +1 because of <EOS> in each synonyms                
                    syn += [params['mappings']['word_map']['<PAD>']] * pad_syn
                span_syn[idx2] = syn
            # pad the number of synonym in each span
            pad_syn = number_of_syns - len(span_syn)
            if pad_syn > 0:
                span_syn += [[params['mappings']['word_map']['<PAD>']]* (number_of_syn_tokens + 1)] * pad_syn
                span_syn_l += [0] * pad_syn
            elif pad_syn < 0:
                span_syn = span_syn[:number_of_syns]
                span_syn_l = span_syn_l[:number_of_syns]
            
            assert len(span_syn) == number_of_syns            
            assert len(span_s) == params["max_span_width"] + 1, span_s
            assert len(span_t) == params["max_span_width"] + 1

            span_source[idx1] = span_s
            span_target[idx1] = span_t
            span_length[idx1] = span_l
            span_syns[idx1] = np.array(span_syn)
            span_syn_lens [idx1] = np.array(span_syn_l)
            
        #pad the number of spans in each sentence
        span_source += [[params['mappings']['word_map']['<PAD>']] * (params["max_span_width"]+ 1)] * num_padding_spans
        span_target += [[params['mappings']['word_map']['<PAD>']] * (params["max_span_width"]+ 1)] * num_padding_spans
        span_length += [0]*num_padding_spans
        if num_padding_spans > 0:
            span_syns += [np.array([[params['mappings']['word_map']['<PAD>']]* (number_of_syn_tokens+1)] * number_of_syns)] * num_padding_spans
            span_syn_lens += [np.array([0] * number_of_syns)] * num_padding_spans
        
        assert len(span_source) == max_span_labels
        assert len(span_target) == max_span_labels
        assert len(span_length) == max_span_labels
        assert len(span_syns) == max_span_labels
        assert len(span_syn_lens) == max_span_labels
        
    return max_span_labels



def group_params(param_optimizers, params):
    return gen_grouped_ner_vae_params(param_optimizers, params)
    
def gen_grouped_ner_vae_params(param_optimizers, params):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    ner_lr = params['ner_learning_rate']
    vae_lr = params['vae_learning_rate']

    optimizer_grouped_parameters = [
        {
            "name": 'ner1',
            "params": [
                p
                for n, p in param_optimizers
                if 'NER_layer' in n and ('bert' in n or 'entity_classifier' in n ) and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": ner_lr
        },
        {
            "name": 'ner2',
            "params": [
                p
                for n, p in param_optimizers
                if 'NER_layer' in n and ('bert' in n or 'entity_classifier' in n ) and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": ner_lr
        },
        {
            "name": 'vae',
            "params": [
                p
                for n, p in param_optimizers
                if 'NER_layer' in n and not 'bert' in n and not 'entity_classifier' in n 
            ],
            # "weight_decay": 0.0001,
            "weight_decay": 0.0,
            "lr": vae_lr
        },
        {
            "name": 'nel',
            "params": [
                p
                for n, p in param_optimizers
                if 'NEL_layer' in n 
            ],
            "weight_decay": 0.0001,
            "lr": vae_lr
        }

    ]

    return optimizer_grouped_parameters


def get_tensors(data_ids, data, params):    

    bert_tokens = [
        data["nn_data"]["bert_tokens"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    token_masks = [
        data["nn_data"]["token_mask"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    attention_masks = [
        data["nn_data"]["attention_mask"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    span_indices = [
        data["nn_data"]["span_indices"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    span_labels = [
        data["nn_data"]["span_labels"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    entity_masks = [
        data["nn_data"]["entity_masks"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_terms = [
        data["nn_data"]["span_terms"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_sources = [
        data["nn_data"]["span_sources"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_targets = [
        data["nn_data"]["span_targets"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_lengths = [
        data["nn_data"]["span_lengths"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_synonyms = [
        data["nn_data"]["span_synonyms"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_syn_lengths = [
        data["nn_data"]["span_syn_lengths"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    bert_tokens = copy.deepcopy(bert_tokens)
    token_masks = copy.deepcopy(token_masks)
    attention_masks = copy.deepcopy(attention_masks)
    span_indices = copy.deepcopy(span_indices)
    span_labels = copy.deepcopy(span_labels)    
    entity_masks = copy.deepcopy(entity_masks)
    span_terms = copy.deepcopy(span_terms)
    span_sources = copy.deepcopy(span_sources)
    span_targets = copy.deepcopy(span_targets)
    span_lengths = copy.deepcopy(span_lengths)
    
    
    span_synonyms = copy.deepcopy(span_synonyms)    
    span_syn_lengths = copy.deepcopy(span_syn_lengths)

    # t1 = time()
    max_span_labels = padding_samples(
        bert_tokens,
        token_masks,
        attention_masks,
        span_indices,
        span_labels,
        span_sources,
        span_targets,
        span_lengths,        
        entity_masks,
        span_synonyms, 
        span_syn_lengths,  
        params
    )
    # t2 = time()
    # print("Padding samples: " + _humanized_time(t2-t1) + "\n")

    batch_bert_tokens = torch.tensor(bert_tokens, dtype=torch.long, device=params["device"])

    batch_token_masks = torch.tensor(
        token_masks, dtype=torch.uint8, device=params["device"]
    )
    batch_attention_masks = torch.tensor(
        attention_masks, dtype=torch.long, device=params["device"]
    )
    batch_span_indices = torch.tensor(
        span_indices, dtype=torch.long, device=params["device"]
    )
    batch_span_labels = torch.tensor(
        span_labels, dtype=torch.float, device=params["device"]
    )

    batch_entity_masks = torch.tensor(
        entity_masks, dtype=torch.int8, device=params["device"]
    )

    batch_span_sources = torch.tensor(
        span_sources, dtype=torch.long, device=params["device"]
    )

    batch_span_targets = torch.tensor(
        span_targets, dtype=torch.long, device=params["device"]
    )
    batch_span_lengths = torch.tensor(
        span_lengths, dtype=torch.long, device=params["device"]
    )
   
    batch_span_synonyms = torch.tensor(
        span_synonyms, dtype=torch.long, device=params["device"]
    )
    batch_span_syn_lengths = torch.tensor(
        span_syn_lengths, dtype=torch.long, device=params["device"]
    )

    return (        
        batch_bert_tokens,
        batch_token_masks,
        batch_attention_masks,
        batch_span_indices,
        batch_span_labels,
        batch_entity_masks,        
        span_terms,
        batch_span_sources,
        batch_span_targets,
        batch_span_lengths,                
        max_span_labels,
        batch_span_synonyms,
        batch_span_syn_lengths        
    )


def save_best_fscore(current_params, last_params):
    # This means that we skip epochs having fscore <= previous fscore
    return current_params["fscore"] <= last_params["fscore"]


def save_best_loss(current_params, last_params):
    # This means that we skip epochs having loss >= previous loss
    return current_params["loss"] >= last_params["loss"]


def handle_checkpoints(
        model,
        checkpoint_dir,
        resume=False,
        params={},
        filter_func=None,        
        num_saved=-1,
        filename_fmt="${filename}_${epoch}_${fscore}.pt",
):
    if resume:
        # List all checkpoints in the directory
        checkpoint_files = sorted(
            glob(os.path.join(checkpoint_dir, "*.*")), reverse=True
        )

        # There is no checkpoint to resume
        if len(checkpoint_files) == 0:
            return None

        last_checkpoint = None

        if isinstance(resume, dict):
            for previous_checkpoint_file in checkpoint_files:
                previous_checkpoint = torch.load(previous_checkpoint_file, map_location=params['device'])
                previous_params = previous_checkpoint["params"]
                if all(previous_params[k] == v for k, v in resume.items()):
                    last_checkpoint = previous_checkpoint
        else:
            # Load the last checkpoint for comparison
            last_checkpoint = torch.load(checkpoint_files[0], map_location=params['device'])

        print(checkpoint_files[0])

        # There is no appropriate checkpoint to resume
        if last_checkpoint is None:
            return None

        print('Loading model from checkpoint', checkpoint_dir)

        # Restore parameters
        current_model_dict = model.state_dict()
        new_state_dict = {k:v if v.size()==current_model_dict[k].size()  
                            else  
                                current_model_dict[k] for k,v in zip(current_model_dict.keys(), last_checkpoint["model"].values())}
        model.load_state_dict(new_state_dict)
        return last_checkpoint["params"]
    else:
        # Validate params
        varname_pattern = re.compile(r"\${([^}]+)}")
        for varname in varname_pattern.findall(filename_fmt):
            assert varname in params, (
                    "Params must include variable '%s'" % varname
            )

        # Create a new directory to store checkpoints if not exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Make the checkpoint unique
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

        # Store the current status
        random_states = {}
        random_states["random_state"] = random.getstate()
        random_states["np_random_state"] = np.random.get_state()
        random_states["torch_random_state"] = torch.get_rng_state()

        for device_id in range(torch.cuda.device_count()):
            random_states[
                "cuda_random_state_" + str(device_id)
                ] = torch.cuda.get_rng_state(device=device_id)

        # List all checkpoints in the directory
        checkpoint_files = sorted(
            glob(os.path.join(checkpoint_dir, "*.*")), reverse=True
        )

        # Now, we can define filter_func to save the best model
        # temporarily close it to test the saved model
        if filter_func and len(checkpoint_files):
            # Load the last checkpoint for comparison
            last_checkpoint = torch.load(checkpoint_files[0], map_location=params['device'])

            if timestamp <= last_checkpoint["timestamp"] or filter_func(
                    params, last_checkpoint["params"]
            ):
                return None

        checkpoint_file = (
                timestamp  # For sorting easily
                + "_"
                + varname_pattern.sub(
            lambda m: str(params[m.group(1)]), filename_fmt
        )
        )
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)

        # In case of using DataParallel
        model = model.module if hasattr(model, "module") else model

        print("***** Saving model *****")

        # Save the new checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "random_states": random_states,
                "params": params,
                "timestamp": timestamp,
            },
            checkpoint_file,
        )       
        print("Saved checkpoint as `%s`" % checkpoint_file)

        # Remove old checkpoints
        if num_saved > 0:
            for old_checkpoint_file in checkpoint_files[num_saved - 1:]:
                os.remove(old_checkpoint_file)

        is_save = True

        return is_save


def get_saved_epoch(model_dir):
    st_ep = 1
    checkpoint_files = sorted(
        glob(os.path.join(model_dir, "*.*")), reverse=True
    )

    if len(checkpoint_files) > 0:
        saved_model_fname = os.path.basename(checkpoint_files[0])
        saved_ep = saved_model_fname.split(".")[0].split("_")[-2]
        if saved_ep.isdigit():
            st_ep = int(saved_ep)
            if st_ep < 1:
                st_ep = 1

    return st_ep


def load_synonyms(fileIn):
    syns = {}
    with open(fileIn) as file:
        for line in file:
            cols = line.rstrip().split('\t')
            if len(cols) < 2:
                continue
            syn_text = []
            for text in cols[1:]:
                if text != cols[0] and text not in syn_text: #skip the span itself as a snynonyms
                    syn_text.append(text)
            if len(syn_text) > 0:
                syns[cols[0]] = syn_text
    return syns

def abs_path(*paths):
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, *paths)
    )


def read_lines(filename):
    with open(abs_path(filename), "r", encoding="UTF-8") as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n"):
    is_first_line = True
    # make_dirs(os.path.dirname(filename))
    # os.makedirs(filename)
    # with open(abs_path(filename), "w", encoding="UTF-8") as f:
    with open(filename, "w", encoding="UTF-8") as f:
        for line in lines:
            if is_first_line:
                is_first_line = False
            else:
                f.write(linesep)
            f.write(line)

        # fig bug that not write file with empty prediction
        # if len(lines) == 0:
        #     print(filename)
        #     f.write(linesep)



def write_annotation_file(
        ann_file, entities=None, triggers=None, relations=None, events=None
):
    lines = []

    def annotate_text_bound(entities):
        for entity in entities.values():
            entity_annotation = "{}\t{} {} {}\t{}".format(
                entity["id"],
                entity["type"],
                entity["start"],
                entity["end"],
                entity["ref"],
            )
            lines.append(entity_annotation)

    if entities:
        annotate_text_bound(entities)

    if triggers:
        annotate_text_bound(triggers)

    if relations:
        for relation in relations.values():
            relation_annotation = "{}\t{} {}:{} {}:{}".format(
                relation["id"],
                relation["role"],
                relation["left_arg"]["label"],
                relation["left_arg"]["id"],
                relation["right_arg"]["label"],
                relation["right_arg"]["id"],
            )
            lines.append(relation_annotation)

    if events:
        for event in events.values():
            event_annotation = "{}\t{}:{}".format(
                event["id"], event["trigger_type"], event["trigger_id"]
            )
            for arg in event["args"]:
                event_annotation += " {}:{}".format(arg["role"], arg["id"])
            lines.append(event_annotation)

    write_lines(lines, ann_file)


def text_decode(sens, tokenizer):
    """Decode text from subword indices using pretrained bert"""

    ids = [id.item() for id in sens]
    orig_text = tokenizer.decode(ids, skip_special_tokens=True)

    return orig_text

def load_pretrained_embeds(pret_embeds_file, embedding_dim):
    pretrained = {}
    with open(pret_embeds_file, 'r') as infile:
        for line in infile:
            line = line.rstrip().split(' ')
            word, vec = line[0], list(map(float, line[1:]))
            if (word not in pretrained) and (len(vec) == embedding_dim):
                pretrained[word] = np.asarray(vec, 'f')
    print('Loaded {}, {}-dimensional pretrained word-embeddings\n'.format(len(pretrained), embedding_dim))
    return pretrained

def load_bert_weights(args):
    ## Encoder   
    tokenizer_encoder = AutoTokenizer.from_pretrained(args['bert_model'])
    if args['block_size'] <= 0:
        args['block_size'] = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args['block_size'] = min(args['block_size'], tokenizer_encoder.max_len_single_sentence) 

    return tokenizer_encoder 