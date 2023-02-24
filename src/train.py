import os
import random
import time
import numpy as np
import torch
import pickle
from tqdm import tqdm, trange

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from transformers import AdamW, get_linear_schedule_with_warmup
from eval.evaluation import eval
from loader.prepData import prepdata
from loader.prepNN import mapping
from loader.prepNN import prep4nn
from model import SpanNER
from utils import utils

from utils.utils import (debug,
    extract_scores,    
)


def main():
    # check running time
    t_start = time.time()

    # set config path by command line
    inp_args = utils._parsing()
    config_path = inp_args['yaml']

    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    parameters.update(inp_args)        
    parameters['sample_size'] = 5 # Defines the number of samples for the ib method.
    if parameters['vib']:                
        parameters['activation'] = 'relu'
        parameters['kl_annealing'] = 'linear'
    parameters['ner_learning_rate'] = float(parameters['ner_learning_rate'])
    parameters['vae_learning_rate'] = float(parameters['vae_learning_rate'])
    
    if parameters['gpu'] >= 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(parameters['gpu'])
        parameters['n_gpu'] = 1
    else:
        device = torch.device("cpu")

  
    print('device', device)

    parameters['device'] = device    

    # Fix seed for reproducibility
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(parameters['seed'])
    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)

    # Force predict = False   
    parameters['predict'] = False
    if parameters['span_reco'] or parameters['syn_gen'] or parameters['span_syn']:
        parameters['ner_vae_joint'] = True
    else:
        parameters['ner_vae_joint'] = False

    # print config
    utils._print_config(parameters, config_path)

    tokenizer_vae_encoder = utils.load_bert_weights(parameters)        
    if parameters['ner_vae_joint']:        
        parameters['pretrained_wordvec'] = utils.load_pretrained_embeds(parameters['pretrained_wordvec'], parameters['word_embed_dim'])
        
    train_data = prepdata.prep_input_data(parameters['train_data'], parameters)
    dev_data = prepdata.prep_input_data(parameters['dev_data'], parameters)
    test_data = prepdata.prep_input_data(parameters['test_data'], parameters)
    # mapping
    parameters = mapping.generate_map(train_data, dev_data, test_data, parameters) 
    # nner:
    parameters['mappings']['nn_mapping'] = utils.gen_nn_mapping(parameters['mappings']['tag_map'],
                                                                parameters['mappings']['tag2type_map'],
                                                                parameters['trTypes_Ids'])

    train, train_events_map = prep4nn.data2network(train_data, 'train', tokenizer_vae_encoder, parameters)
    dev, dev_events_map = prep4nn.data2network(dev_data, 'demo', tokenizer_vae_encoder, parameters)
    if len(train) == 0:
        raise ValueError("Train set empty.")
    if len(dev) == 0:
        raise ValueError("Test set empty.")

    if parameters['syn_gen'] == True or parameters['span_syn'] == True:
        #load the synonyms of spans for the training set
        train_syns = utils.load_synonyms('data/' + parameters['task_name'] + '/' + parameters['task_name'] + '_train_synExact.txt')
        dev_syns = utils.load_synonyms('data/' + parameters['task_name'] + '/' + parameters['task_name'] + '_dev_synExact.txt')
    
    train_data = prep4nn.torch_data_2_network(cdata2network=train, tokenizer_encoder=tokenizer_vae_encoder,
                                                  events_map=train_events_map,
                                                  params=parameters,
                                                  do_get_nn_data=True,
                                                  synSearch=train_syns)
    dev_data = prep4nn.torch_data_2_network(cdata2network=dev, tokenizer_encoder=tokenizer_vae_encoder,                                                                                               
                                                events_map=dev_events_map,
                                                params=parameters,
                                                do_get_nn_data=True,
                                                synSearch=dev_syns)
    trn_data_size = len(train_data['nn_data']['bert_tokens'])
    dev_data_size = len(dev_data['nn_data']['bert_tokens'])

    parameters['dev_data_size'] = dev_data_size

    train_data_ids = TensorDataset(torch.arange(trn_data_size))
    dev_data_ids = TensorDataset(torch.arange(dev_data_size))

    # shuffle
    train_sampler = RandomSampler(train_data_ids)
    train_dataloader = DataLoader(train_data_ids, sampler=train_sampler, batch_size=parameters['batchsize'])
    dev_sampler = SequentialSampler(dev_data_ids)
    dev_dataloader = DataLoader(dev_data_ids, sampler=dev_sampler, batch_size=parameters['batchsize'])

    # 2. model
    model = SpanNER.SpanNER(parameters)

    if parameters['start_epoch'] > 0:
        utils.handle_checkpoints(model=model,
                checkpoint_dir=parameters['joint_model_dir'],
                params={
                    'device': device
                },
                resume=True)
       
    # 3. optimizer
    assert (
            parameters['gradient_accumulation_steps'] >= 1
    ), "Invalid gradient_accumulation_steps parameter, should be >= 1."

    parameters['batchsize'] //= parameters['gradient_accumulation_steps']

    num_train_steps = parameters['epoch'] * (
            (trn_data_size - 1) // (parameters['batchsize'] * parameters['gradient_accumulation_steps']) + 1)
    parameters['voc_sizes']['num_train_steps'] = num_train_steps

    parameters['x0'] = int(num_train_steps/2)

    model.to(device)

    # Prepare optimizer
    t_total = num_train_steps
    optimizer_grouped_parameters = utils.group_params(list(model.named_parameters()), parameters)    
    optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=parameters['ner_learning_rate'],
            correct_bias=False)
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=parameters['warmup_proportion']*t_total, num_training_steps=t_total) 
            
    # 4. training       
    train_func(train_data_loader=train_dataloader, dev_data_loader=dev_dataloader,
                       train_data=train_data, dev_data=dev_data, params=parameters, model=model,
                       optimizer=optimizer, 
                       scheduler=scheduler)
       

    print('TRAINING: DONE!')

    # calculate running time
    t_end = time.time()
    print('TOTAL RUNNING TIME: {}'.format(utils._humanized_time(t_end - t_start)))


def train_func(
        train_data_loader,
        dev_data_loader,
        train_data,
        dev_data,
        params,
        model,
        optimizer,
        scheduler=None,        
        tb_writer=None,
):

    is_params_saved = False
    global_steps = 0

    gradient_accumulation_steps = params["gradient_accumulation_steps"]

    ner_prf_dev_str, ner_prf_dev_sof = [], []

    tr_batch_losses_ = []

    # create output directory for results
    result_dir = params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save params:
    if params['save_params']:
        if not is_params_saved:
            # saved_params_path = result_dir + params['task_name'] + '.param'
            saved_params_path = params['params_dir']
            with open(saved_params_path, "wb") as f:
                pickle.dump(params, f)
            print('SAVED PARAMETERS!')

    st_ep = 0

    # n_iter = int(params['epoch']) * len(train_data_loader) 
    n_iter = params['voc_sizes']['num_train_steps']  # number of train steps from main.py    
    model.zero_grad()

    kl_w = kl_anneal_function('logistic', n_iter,x0=params['x0'])
    
    params['best_epoch'] = 0
    scores = [0]
    total_steps = 0
    training_type = 'train'    
    for epoch in trange(st_ep, int(params["epoch"]), desc="Epoch"):        
        if epoch < params['start_epoch']:
            continue

        if epoch < params['ner_start_epoch']:
            training_type = 'pretrain'            
        else:
            training_type = 'train'

        model.train()
        tr_loss = 0
        ner_loss = gen_loss = 0        
        nb_tr_steps = 0
        reco_loss = kld_gen = kld_org = 0
        print(
            "====================================================================================================================")
        # print()
        debug(f"[1] Epoch: {epoch + 1}\n")

        # for mini-batches
        for step, batch in enumerate(
                tqdm(train_data_loader, desc="Iteration", leave=False)
        ):           
            tr_data_ids = batch
            # e_start = time.time()
            tensors = utils.get_tensors(tr_data_ids, train_data, params)           
            
            total_loss, results_loss, _ = model(tensors, epoch=epoch, kl_w=kl_w[nb_tr_steps],stage=training_type)
            
            if gradient_accumulation_steps > 1:
                total_loss /= gradient_accumulation_steps

            tr_loss += total_loss.item()
            nb_tr_steps += 1            
            ner_loss += results_loss['ner_loss']           
            total_steps += 1
            if params['ner_vae_joint']:          
                reco_loss += results_loss['reco_loss'] * (1 - params['task_weight'])
                gen_loss += results_loss['gen_loss']  * (1 - params['task_weight'])
                kld_org += results_loss['kld_reco']                
                kld_gen += results_loss['kld_gen']
          
            try:
                total_loss.backward()
            except RuntimeError as err:
                print('\nIn code: RuntimeError loss.backward(): ', err)
                return

            if (step + 1) % params["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_steps += 1

                # Clear GPU unused RAM:
                if params['gpu'] >= 0:
                    torch.cuda.empty_cache()

        # save for batches
        ep_loss = tr_loss / nb_tr_steps
        ner_loss = ner_loss / nb_tr_steps        
        gen_loss = gen_loss / nb_tr_steps
        reco_loss = reco_loss / nb_tr_steps
        kld_org = kld_org / nb_tr_steps
        kld_gen = kld_gen / nb_tr_steps
        tr_batch_losses_.append(float("{0:.2f}".format(ep_loss)))
        # print()
        debug(f"[2] Train loss: {ep_loss}\n")
        debug(f"[3] Global steps: {global_steps}\n")
        if tb_writer!=None:        
            tb_writer.log({'ner_loss': ner_loss, 'reco_loss': reco_loss, 
                            'gen_loss': gen_loss, 'total_loss': ep_loss}, commit=False, step=epoch)
     
        if (training_type == 'pretrain' and epoch + 1 == params['ner_start_epoch'] - 1) \
            or training_type == 'train':
            print("+" * 10 + "RUN EVALUATION" + "+" * 10)
            all_scores = eval(
                model=model,
                eval_dir=params['dev_data'],
                result_dir=result_dir,
                eval_dataloader=dev_data_loader,
                eval_data=dev_data,                
                params=params,
                epoch=epoch,                
            )

            if len(all_scores) > 0:
                # show scores                
                if 'spanBleu' in all_scores:
                    if training_type == 'pretrain':
                        scores.append(all_scores['spanBleu'])
                    if tb_writer!=None:
                        tb_writer.log({'eval/spanBleu' : all_scores['spanBleu']}, step=epoch)
                
                if 'entBleu' in all_scores:
                    if training_type == 'pretrain':
                        scores.append(all_scores['entBleu'])
                    if tb_writer!=None:
                        # tb_writer.add_scalar('eval/entBleu', all_scores['entBleu'], epoch)
                        tb_writer.log({'eval/entBleu':all_scores['entBleu']}, step=epoch)
                        

                if 'synBleu' in all_scores:
                    if training_type == 'pretrain':
                        scores.append(all_scores['synBleu'])
                    if tb_writer!=None:
                        # tb_writer.add_scalar('eval/synBleu', all_scores['synBleu'], epoch)
                        tb_writer.log({'eval/synBleu':all_scores['synBleu']}, step=epoch)

                if 'ner' in all_scores:                
                    show_results(all_scores['ner'], ner_prf_dev_str, ner_prf_dev_sof)
                    if tb_writer!=None:
                        # tb_writer.add_scalar('eval/f1-score',all_scores['ner']['micro']['st_f'],epoch)
                        tb_writer.log({'eval/f1-score':all_scores['ner']['micro']['st_f']}, step=epoch)
                    scores.append(all_scores['ner']['micro']['st_f'])
                    if max(scores) <= all_scores['ner']['micro']['st_f']:
                        params['best_epoch'] = epoch
                        

        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()
        
    return max(scores)

def show_results(n2c2_scores, ner_prf_dev_str, ner_prf_dev_sof):
    ner_prf_dev_str.append(
        [
            float("{0:.2f}".format(n2c2_scores['micro']['st_p'])),
            float("{0:.2f}".format(n2c2_scores['micro']['st_r'])),
            float("{0:.2f}".format(n2c2_scores['micro']['st_f'])),
        ]
    )

    # tb_writer.add_scalar('f1-score', n2c2_scores['NER']['micro']['st_f'])
    ner_prf_dev_sof.append(
        [
            float("{0:.2f}".format(n2c2_scores['micro']['so_p'])),
            float("{0:.2f}".format(n2c2_scores['micro']['so_r'])),
            float("{0:.2f}".format(n2c2_scores['micro']['so_f'])),
        ]
    )

    # ner
    _ = extract_scores('n2c2 ner strict (micro)', ner_prf_dev_str)
    extract_scores('n2c2 ner soft (micro)', ner_prf_dev_sof)

def kl_anneal_function(anneal_function, steps, k=0.0025, x0=2500, beta=0.5):
    """ Credits to: https://github.com/timbmg/Sentence-VAE/blob/master/train.py#L69 """
    if anneal_function == 'logistic':
        return [float(1 / (1 + np.exp(-k * (step - x0)))) for step in range(steps)]
    elif anneal_function == 'linear':
        return [min(1, int(step / x0)) for step in range(steps)]
    else: #constant 
        return [beta] * steps


if __name__ == '__main__':
    main()