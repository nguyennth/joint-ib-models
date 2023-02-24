from collections import defaultdict
import enum
import time
import os
import torch
from tqdm import tqdm


from utils import utils
from utils.utils import _humanized_time, write_annotation_file
from scripts.plots_tsne import plot_tsne_2d

from torchtext.data.metrics import bleu_score

def eval(model, eval_dir, result_dir, eval_dataloader, eval_data,
            params, epoch=0):
    mapping_id_tag = params['mappings']['nn_mapping']['id_tag_mapping']
    fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []
    
    ent_anns = []

    # Evaluation phase
    model.eval()
    # nner
    all_ner_preds, all_ner_golds, all_ner_terms = [], [], []

    t_start = time.time()

    posteriors_z = []
    posteriors_label = []    
    gold_embeddings = []
    vib_posteriors = []
    vib_embeddings = []
    
    original_spans = [] #orginal spans
    generated_spans = []  #generated spans    
    org_syns = [] #original syns
    generated_syns = [] #generated sysn
    generated_entities = [] #generated entities
    original_entities = [] #original entities
    actual_labels = []
    predict_labels = []


    for step, batch in enumerate(
            tqdm(eval_dataloader, desc="Iteration", leave=False)
    ):
        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, eval_data, params)
        nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, \
        nn_span_labels, nn_entity_masks, _, \
        span_sources, span_targets, span_lengths, _ , \
        span_synonyms, span_syn_lengths = tensors
        
        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        offsets = [
            eval_data["offsets"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        words = [
            eval_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        sub_to_words = [
            eval_data["sub_to_words"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        subwords = [
            eval_data["subwords"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
       
        with torch.no_grad():
            _, _, ner_out = model(tensors, stage='test',sampling_type='argmax')            
            #generate spans and synonyms for evaluation        
            if ner_out['latent_z'] != None:                             
                for id, z in enumerate(ner_out['latent_z']):                        
                    z = torch.tensor(z, dtype=torch.float, device=params["device"]).unsqueeze(0)
                    temp = model.NER_layer.inference(z, 'span')
                    # generated_spans.append(temp)
                    # for label in e_golds[id]:
                        # if label == 0: continue
                    generated_entities.append(temp)
            
            if ner_out['latent_z_syn'] != None:
                for id, z in enumerate(ner_out['latent_z_syn']):
                    z = torch.tensor(z, dtype=torch.float, device=params["device"]).unsqueeze(0)
                    temp = model.NER_layer.inference(z, 'syn')
                    generated_syns.append(temp)
       
        for sent_term in ner_out['gold_terms']:
            for ent in sent_term.text:
                original_entities.append([sent_term.text[ent]])                
                syns = []
                for syn in sent_term.syns[ent]:
                    toks = syn.split(' ')
                    syns.append(toks)
                if len(syns) > 0: #if there are syns for the target entity
                    org_syns.append(syns)                
                else: #use the entity text as a synonym
                    org_syns.append([sent_term.text[ent]])
        
        if params['span_reco']:
            assert len(generated_entities) == len(original_entities), "Error in generating entities: the number is not the same"  

        if params['syn_gen']:
            assert len(generated_syns) == len(org_syns), "Error in generating synonyms: the number is not the same"

        if params['predict']:
            #print out posterior_z for visualisation       
            if ner_out['latent_z'] != None:
                posteriors_z += ner_out['latent_z'].tolist()    
            if ner_out['vib_z'] != None:
                vib_posteriors += ner_out['vib_z'].tolist()
            if ner_out['vib_emb'] != None:
                vib_embeddings += ner_out['vib_emb'].tolist()
            posteriors_label += ner_out['labels_z']     
            gold_embeddings += ner_out['gold_embs'].tolist()
        
        fidss.append(fids)
        ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                            'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                            'ner_terms': ner_out['terms']}
        ent_anns.append(ent_ann)
        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()

    all_scores = {}    
    if params['ner_vae_joint']:
        spanScore, entScore, synScore = evaluate_bleu(generated_spans, original_spans, generated_entities, 
                original_entities, org_syns, generated_syns, params['result_dir'])
        all_scores['spanBleu'] = spanScore
        all_scores['entBleu'] = entScore
        all_scores['synBleu'] = synScore
   
    if ('preds' in ner_out and epoch >= params['ner_start_epoch']) or params['predict']:        
        n2c2_scores = estimate_ent(ref_dir=eval_dir,
                                result_dir=result_dir,
                                fids=fidss,
                                ent_anns=ent_anns,                              
                                params=params)        
        show_scores(n2c2_scores['NER'])
        all_scores['ner'] = n2c2_scores['NER']

    if params['predict']:            
        if len(gold_embeddings) > 0:
            
            plot_tsne_2d(gold_embeddings, posteriors_label, params['task_name'], params['latent_size'], 
                True,  params['result_dir'], 'gold_emb')
        if len(posteriors_z) > 0:
            plot_tsne_2d(posteriors_z, posteriors_label, params['task_name'], params['latent_size'], 
                True,  params['result_dir'], 'vae_posts')
        if len(vib_posteriors) > 0:
            plot_tsne_2d(vib_posteriors, posteriors_label, params['task_name'], params['latent_size'], 
                True,  params['result_dir'], 'vib_posts')
        if len(vib_embeddings) > 0:
            plot_tsne_2d(vib_embeddings, posteriors_label, params['task_name'], params['latent_size'], 
                True,  params['result_dir'], 'vib_emb')
    else:      # saving models    
        if epoch > params['save_st_ep']:
            save_models(model, params, epoch, all_scores)

    t_end = time.time()
    print('Elapsed time: {}\n'.format(_humanized_time(t_end - t_start)))
    
    return all_scores


def estimate_ent(ref_dir, result_dir, fids, ent_anns, params):
    """Evaluate entity performance using n2c2 script"""

    # generate brat prediction
    gen_annotation_ent(fids, ent_anns, params, result_dir)

    # calculate scores
    pred_dir = ''.join([result_dir, 'predictions/ent-ann/'])
    pred_scores_file = ''.join([result_dir, 'predictions/ent-scores-', params['ner_eval_corpus'], '.txt'])

    # run evaluation, output in the score file
    eval_performance(ref_dir, pred_dir, pred_scores_file, params)

    # extract scores
    scores = extract_fscore(pred_scores_file)

    return scores


def evaluate_bleu(generated_spans, original_spans, generated_entities, original_entities, original_syns, generated_syns, result_dir):    
    spanScore = entScore = synScore = 0
    if len(generated_spans) > 0:
        spanScore = bleu_score(generated_spans, original_spans, max_n=2, weights=[0.5, 0.5]) 
        spanScore4 = bleu_score(generated_spans, original_spans) 
        print("Span BLEU Score: ", spanScore)
        print("Span BLEU-4 Score: ", spanScore4)
        with open(result_dir + "/span_generated.txt", 'w') as file:
            for org, gen in zip (original_spans, generated_spans):
                file.write(' '.join(org[0]) + '\t' + ' '.join(gen) + '\n')
        
    if len(generated_entities) > 0:
        assert len(generated_entities) == len(original_entities), "Error in generating entities"
        entScore = bleu_score(generated_entities, original_entities, max_n=2, weights=[0.5, 0.5])    
        entScore4 = bleu_score(generated_entities, original_entities)    
        print("Entity BLEU Score:", entScore)
        print("Entity BLEU-4 Score:", entScore4)
        with open(result_dir + "/ent_generated.txt", 'w') as file:
            for org, gen in zip(original_entities, generated_entities):
                file.write(' '.join(org[0]) + '\t' + ' '.join(gen) + '\n')
        
    if len(generated_syns) > 0:
        with open (result_dir + '/syn_generated.txt', 'w') as fWrite:
            for id, (source_str, syn_str) in enumerate(zip(original_entities, generated_syns)):
                source_str[0] = [tok.lower() for tok in source_str[0]]                
                syn_str = [tok.lower() for tok in syn_str]
                fWrite.write(' '.join(source_str[0]) + '\t' + ' '.join(syn_str) + '\n')               
                original_entities[id] = source_str
        synScore = bleu_score(generated_syns, original_syns, max_n=2, weights=[0.5, 0.5])
        synScore4 = bleu_score(generated_syns, original_syns)
        print("Synonym BLEU Score:", synScore)
        print("Synonym BLEU-4 Score:", synScore4)
        

    return spanScore, entScore, synScore

def eval_performance(ref_dir, pred_dir, pred_scores_file, params):
    # run evaluation script
    command = ''.join(
        ["python3 ", params['eval_script_path'], " --ner-eval-corpus ", params['ner_eval_corpus'], " ", ref_dir, " ",
         pred_dir, " > ", pred_scores_file])
    os.system(command)


def extract_fscore(path):
    file = open(path, 'r')
    lines = file.readlines()
    report = defaultdict()
    report['NER'] = defaultdict()
    report['REL'] = defaultdict()

    ent_or_rel = ''
    for line in lines:
        if '*' in line and 'TRACK' in line:
            ent_or_rel = 'NER'
        elif '*' in line and 'RELATIONS' in line:
            ent_or_rel = 'REL'
        elif len(line.split()) > 0 and line.split()[0] == 'Overall':
            tokens = line.split()
            if len(tokens) > 8:
                strt_f, strt_r, strt_p, soft_f, soft_r, soft_p \
                    = tokens[-7], tokens[-8], tokens[-9], tokens[-4], tokens[-5], tokens[-6]
            else:
                strt_f, strt_r, strt_p, soft_f, soft_r, soft_p \
                    = tokens[-4], tokens[-5], tokens[-6], tokens[-1], tokens[-2], tokens[-3]
            if line.split()[1] == '(micro)':
                mi_or_mc = 'micro'
            elif line.split()[1] == '(macro)':
                mi_or_mc = 'macro'
            else:
                mi_or_mc = ''
            if mi_or_mc != '':
                report[ent_or_rel][mi_or_mc] = {'st_f': float(strt_f.strip()) * 100,
                                                'st_r': float(strt_r.strip()) * 100,
                                                'st_p': float(strt_p.strip()) * 100,
                                                'so_f': float(soft_f.strip()) * 100,
                                                'so_r': float(soft_r.strip()) * 100,
                                                'so_p': float(soft_p.strip()) * 100}

    return report


def gen_annotation_ent(fidss, ent_anns, params, result_dir):
    """Generate entity and relation prediction"""

    dir2wr = ''.join([result_dir, 'predictions/ent-ann/'])
    if not os.path.exists(dir2wr):
        os.makedirs(dir2wr)
    else:
        os.system('rm ' + dir2wr + '*.ann')

    # Initial ent+rel map
    map = defaultdict()
    for fids in fidss:
        for fid in fids:
            map[fid] = {}

    for xi, (fids, ent_ann) in enumerate(zip(fidss, ent_anns)):
        # Mapping entities
        entity_map = defaultdict()
        for xb, (fid) in enumerate(fids):
            span_indices = ent_ann['span_indices'][xb]
            ner_terms = ent_ann['ner_terms'][xb]
            ner_preds = ent_ann['ner_preds'][xb]
            words = ent_ann['words'][xb]
            offsets = ent_ann['offsets'][xb]
            sub_to_words = ent_ann['sub_to_words'][xb]

            entities = map[fid]
            # e_count = len(entities) + 1

            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1:
                    break
                if ner_preds[x] > 0:
                    # e_id = 'T' + str(e_count)
                    # e_count += 1
                    try:
                        e_id = ner_terms.id2term[x]
                        e_type = params['mappings']['rev_type_map'][
                            params['mappings']['nn_mapping']['tag2type_map'][ner_preds[x]]]
                        e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        # entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                        #     ner_preds[x], e_id, e_type, e_words, e_offset)
                        entity_map[(xb, x)] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error, fid)
        

    for fid, ners in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners)

def convert2words(ids, rev_word_map):
    orig_text = []
    for id in ids:     
        if id.item() == 0: break 
        if id.item() < 3: #skip SOS, EOS, PAD
            continue   
        orig_text.append(rev_word_map[id.item()])
    
    return orig_text

def show_scores(n2c2_scores):
    # print()
    print('-----EVALUATING BY N2C2 SCRIPT (FOR ENT & REL)-----\n')
    # print()
    print('STRICT_MATCHING:\n')
    print_scores('NER', n2c2_scores, 'st')
    # print()
    print('SOFT_MATCHING:\n')
    print_scores('NER', n2c2_scores, 'so')



def print_scores(k, v, stoso):
    print(
        k + "(MICRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} , (MACRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f}\n".format(
            v['micro'][stoso + '_p'], v['micro'][stoso + '_r'], v['micro'][stoso + '_f'],
            v['macro'][stoso + '_p'], v['macro'][stoso + '_r'], v['macro'][stoso + '_f']), end="",
    )
    # print()


def save_models(model, params, epoch, all_scores):
    ner_fscore = 0
    if 'ner' in all_scores:
        ner_fscore = all_scores['ner']['micro']['st_f']

    if epoch < params['ner_start_epoch']:
        best_score = 0
        if len(all_scores) > 0:
            if all_scores['entBleu'] > 0:
                best_score = all_scores['entBleu']
            elif all_scores['synBleu'] > 0:
                best_score = all_scores['synBleu']
            elif all_scores['spanBleu'] > 0:
                best_score = all_scores['spanBleu']        
            
    else:
        best_score = ner_fscore

    is_save = False

    # Save models
    all_model_path = params['joint_model_dir']
    is_save = utils.handle_checkpoints(
            model=model,
            checkpoint_dir=all_model_path,
            params={
                "filename": "joint_ib_ner",
                "epoch": epoch,
                "fscore": best_score,
                'device': params['device'],
                'params_dir': params['params_dir'],
                'result_dir': params['result_dir']
            },            
            filter_func=utils.save_best_fscore,
            num_saved=1
        )
    print("Saved all models")
    
    return is_save


def get_entity_attrs(e_span_indice, words, offsets, sub_to_words):
    e_words = []
    e_offset = [-1, -1]
    curr_word_idx = -1
    for idx in range(e_span_indice[0], e_span_indice[1] + 1):
        if sub_to_words[idx] != curr_word_idx:
            e_words.append(words[sub_to_words[idx]])
            curr_word_idx = sub_to_words[idx]
        if idx == e_span_indice[0]:
            e_offset[0] = offsets[sub_to_words[idx]][0]
        if idx == e_span_indice[1]:
            e_offset[1] = offsets[sub_to_words[idx]][1]
    return ' '.join(e_words), (e_offset[0], e_offset[1])

def get_entity_sw_attrs(e_id, e_span_indice, words, offsets, sub_to_words, subwords, sw_offsets, org_mapping):
    e_words = []
    e_offset = [-1, -1]
    sw_text = []
    sw_offset = [-1, -1]

    curr_word_idx = -1
    for idx in range(e_span_indice[0], e_span_indice[1] + 1):
        if sub_to_words[idx] != curr_word_idx:
            e_words.append(words[sub_to_words[idx]])
            curr_word_idx = sub_to_words[idx]
        sw_text.append(subwords[idx])
        if idx == e_span_indice[0]:
            e_offset[0] = offsets[sub_to_words[idx]][0]
            sw_offset[0] = sw_offsets[idx][0]
        if idx == e_span_indice[1]:
            e_offset[1] = offsets[sub_to_words[idx]][1]
            sw_offset[1] = sw_offsets[idx][1]
    org_mapping[e_id] = (' '.join(e_words), (e_offset[0], e_offset[1]))
    return ' '.join(sw_text), (sw_offset[0], sw_offset[1])

def gen_sw_offsets(word_offsets, words, subwords, sub_to_words):
    sw_offsets = []
    last_sw_offsets = -1
    for sw_id, w_id in sub_to_words.items():
        subword = subwords[sw_id].replace('##', '')
        word = words[w_id]
        word_offset = word_offsets[w_id]
        sw_idx = word.index(subword,
                            0 if (last_sw_offsets == -1 or last_sw_offsets < word_offset[0]) else last_sw_offsets - 1 -
                                                                                                  word_offset[0])
        sw_offsets.append((word_offset[0] + sw_idx, word_offset[0] + sw_idx + len(subword)))
        last_sw_offsets = word_offset[0] + sw_idx + len(subword)
    return sw_offsets