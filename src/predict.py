import os
import random
import pickle
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import (BertConfig, BertModel, BertTokenizer,)

from loader.prepNN import prep4nn
from loader.prepData import prepdata
from model import SpanNER
from eval.evaluation import eval
from utils import utils

def set_params(saved_params, pred_params):

    parameters = saved_params

    # overwrite test params to saved dparams
    for param in pred_params:
        parameters[param] = pred_params[param]

    # Fix seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(parameters['seed'])
    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if parameters['gpu'] >= 0:
        device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")

        torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")

    print('device', device)
    parameters['device'] = device

    return parameters



def read_test_data(parameters, tokenizer_vae_encoder):
    test_data = prepdata.prep_input_data(parameters['test_data'], parameters)
    test, test_events_map = prep4nn.data2network(test_data, 'test', tokenizer_vae_encoder, parameters)
    synTest = None
    if parameters['syn_gen'] == True or parameters['span_syn']:
        #load the synonyms of spans for the training set
        synTest = utils.load_synonyms('data/' + parameters['task_name'] + '/' + parameters['task_name'] + '_test_synExact.txt')
        
    test_data = prep4nn.torch_data_2_network(cdata2network=test, tokenizer_encoder=tokenizer_vae_encoder,                                             
                                             events_map=test_events_map,
                                             params=parameters,
                                             do_get_nn_data=True,
                                             synSearch=synTest)

    test_data_size = len(test_data['nn_data']['bert_tokens'])
    test_data_ids = TensorDataset(torch.arange(test_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=parameters['batchsize'])

    return test_data, test_dataloader


def test():
    # check running time
    t_start = time.time()

    # set config path by command line, some configs will be overwritten by the trained configs
    inp_args = utils._parsing()
    config_path = inp_args['yaml']
    
    # load params
    with open(config_path, 'r') as stream:
        pred_params = utils._ordered_load(stream)
    
    pred_params['gpu'] = inp_args['gpu']

    if pred_params['gpu'] >= 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        torch.cuda.set_device(pred_params['gpu'])
    else:
        device = torch.device("cpu")

    pred_params.update(inp_args)

    with open(pred_params['params_dir'], "rb") as f:
        saved_params = pickle.load(f)

    # set params based on the training params
    parameters = set_params(saved_params, pred_params)
    
    parameters['device'] = device
    
    # bert tokenizer weights
    tokenizer_vae_encoder = utils.load_bert_weights(parameters)

    # read data
    test_data, test_dataloader = read_test_data(parameters, tokenizer_vae_encoder)

    # load model
    model = SpanNER.SpanNER(parameters)

    checkpoint_dir = parameters['joint_model_dir']
    
    utils.handle_checkpoints(model=model,
                             checkpoint_dir=checkpoint_dir,
                             params={
                                 'device': parameters['device']
                             },
                             resume=True)

    model.to(parameters['device'])

    if not os.path.exists(parameters['result_dir']):
        os.makedirs(parameters['result_dir'])
    
    eval(
        model=model,
        eval_dir=parameters['test_data'],
        result_dir=parameters['result_dir'],
        eval_dataloader=test_dataloader,
        eval_data=test_data,         
        params=parameters)

    print('PREDICT: DONE!')    
    return


if __name__ == '__main__':
    test()
