"""Read brat format input files."""

import glob
import collections
from collections import OrderedDict
import os


def brat_loader(files_fold, params):
    file_list = glob.glob(files_fold + '*' + '.txt')

    triggers = OrderedDict()    
    entities = OrderedDict()    
    relations = OrderedDict()
    sentences = OrderedDict()

    count = 0
    for filef in sorted(file_list):       
        # count += 1
        # if count % 5 == 0:
        #     break
        
        if filef.split("/")[-1].startswith("."):
            continue
        filename = filef.split('/')[-1].split('.txt')[0]
        ffolder = '/'.join(filef.split('/')[:-1]) + '/'

        # store data for each document
        ftriggers = OrderedDict()
        fentities = OrderedDict()
        # fnorms = OrderedDict()
        frelations = OrderedDict()
        # fevents = OrderedDict()

        idsTR = []
        typesTR = []
        infoTR = OrderedDict()
        termsTR = []

        idsT = []
        typesT = []
        infoT = OrderedDict()
        termsT = []

        idsR = []
        typesR = []
        infoR = OrderedDict()
        
        if os.path.exists(os.path.join(ffolder, "".join([filename, ".ann"]))):
            with open(os.path.join(ffolder, "".join([filename, ".ann"])), encoding="UTF-8") as infile:
                for line in infile:

                    if line.startswith('TR'):
                        line = line.rstrip().split('\t')
                        trId = line[0]
                        tr1 = line[1].split()
                        trType = tr1[0]
                        pos1 = tr1[1]
                        pos2 = tr1[2]
                        text = line[2]

                        idsTR.append(trId)
                        typesTR.append(trType)
                        trigger_info = OrderedDict()
                        trigger_info['id'] = trId
                        trigger_info['type'] = trType
                        trigger_info['pos1'] = pos1
                        trigger_info['pos2'] = pos2
                        trigger_info['text'] = text
                        infoTR[trId] = trigger_info
                        termsTR.append([trId, trType, pos1, pos2, text])

                    elif line.startswith('T'):
                        line = line.rstrip().split('\t')
                        eid = line[0]                       
                        e1 = line[1].split()
                        etype = e1[0]
                        pos1 = e1[1]
                        pos2 = e1[2]
                        text = line[2]

                        idsT.append(eid)
                        typesT.append(etype)
                        ent_info = OrderedDict()
                        ent_info['id'] = eid
                        ent_info['type'] = etype
                        ent_info['pos1'] = pos1
                        ent_info['pos2'] = pos2
                        ent_info['text'] = text
                        infoT[eid] = ent_info
                        termsT.append([eid, etype, pos1, pos2, text])
                    elif line.startswith('R'):
                        line = line.rstrip().split('\t')
                        idR = line[0]
                        typeR = line[1].split()[0]
                        typeR = ''.join([i for i in typeR if not i.isdigit()])
                        args = line[1].split()[1:]
                        arg1id = args[0].split(':')[1]
                        arg2id = args[1].split(':')[1]

                        trig2 = False
                        trig1 = False
                        if arg1id.startswith('TR') and arg2id.startswith('TR'):
                            trig2 = True
                            trig1 = True
                        elif arg1id.startswith('TR'):
                            trig1 = True

                        r_info = OrderedDict()
                        r_info['id'] = idR
                        r_info['type'] = typeR
                        r_info['arg1id'] = arg1id
                        r_info['arg2id'] = arg2id
                        r_info['2trigger'] = trig2
                        r_info['1trigger'] = trig1

                        idsR.append(idR)
                        typesR.append(typeR)
                        infoR[idR] = r_info

                typesTR2 = dict(collections.Counter(typesTR))
                typesT2 = dict(collections.Counter(typesT))
                typesR2 = dict(collections.Counter(typesR))

                ftriggers['data'] = infoTR
                ftriggers['types'] = typesTR
                ftriggers['counted_types'] = typesTR2
                ftriggers['ids'] = idsTR
                ftriggers['terms'] = termsTR

                fentities['data'] = infoT
                fentities['types'] = typesT
                fentities['counted_types'] = typesT2
                fentities['ids'] = idsT
                fentities['terms'] = termsT
                frelations['data'] = infoR
                frelations['types'] = typesR
                frelations['ids'] = idsR
                frelations['counted_types'] = typesR2

        entities[filename] = fentities
        triggers[filename] = ftriggers
        relations[filename] = frelations
        lowerc = params['lowercase']
        with open(ffolder + filename + '.txt', encoding="UTF-8") as infile:
            lines = []
            for line in infile:
                line = line.strip()
                if len(line) > 0:
                    if lowerc:
                        line = line.lower()
                    lines.append(line)
            sentences[filename] = lines

    return triggers, entities, relations, sentences
