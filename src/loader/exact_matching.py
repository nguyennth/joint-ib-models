"""
Exact matching with UMLS to find synonyms
"""

import argparse
import os

class multiSearch(object):
    def __init__(self, fileUMLS):        
        self.umls_dictionary, self.concept_dictionary = self.load_umls(fileUMLS)

    def load_umls(self, file_in):
        """
        load umls concepts into a dictionary in which each mention is a key and its corresponding CUIs is the value
        file_in: a processed file of UMLS in which each line has one term and its corresponding CUIs separated by a tab character

        return:
            - concepts: a dictionary in which the key is a concept id, the value is a list of mentions in that concept
            - mentions: a dictionary in which the key is a mention, the value is a list of concept ids
        """
        concepts = {} 
        mentions = {}
        print("Loading UMLS ...")
        with open(file_in) as file:
            for id, line in enumerate(file):
                # if id == 10: 
                #     break
                items = line.strip().split('\t')                
                mention = items[0]
                if len(mention) < 2: continue                
                for id in items[1:]:
                    if mention in mentions:
                        mentions[mention].append(id)
                    else:
                        mentions[mention] = [id]  
                    
                    if id in concepts:
                        concepts[id].append(mention)
                    else:
                        concepts[id] = [mention]       
                    
                   
        return mentions, concepts

    def search_synonyms(self, fileIn, fileOut):
        #just do exact matching against UMLS
        with open(fileIn) as fileRead:
            texts = {}
            for line in fileRead:
                line = line.rstrip().lower()
                if line not in texts:
                    texts[line] = 1        
        with open(fileOut, 'w') as fileWrite:
            count = 0;
            for line in texts: 
                if line in self.umls_dictionary:
                    concept_ids = self.umls_dictionary[line]
                    results = self.concept_dictionary[concept_ids[0]]   
                    fileWrite.write(line + '\t' + '\t'.join(results) + '\n')                    
                count += 1
                if count % 1000 == 0:
                    print ('Processing ', count)
                    # print('\n')
    
    def check_coverage(self, dirIn):
        findSyn = 0
        total = 0
        for file in os.listdir(dirIn):
            if 'ann' not in file:
                continue
            with open(os.path.join(dirIn,file)) as file:
                for line in file:
                    if line.startswith('T'):
                        total+= 1
                        cols = line.rstrip().split('\t')                        
                        if len(cols) < 3 : continue
                        if cols[2].lower() in self.umls_dictionary:
                            findSyn += 1
        print(dir + '\n')
        print('Total entity: ', total)
        print ('Syn coverage: ', findSyn * 1.0/total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--umls', type=str, help='Path to the UMLS file', default='data/KBs/UMLS/all-umls-synonyms_tok.txt')
    parser.add_argument('--infile',type=str, help='Path to an input file where we have all possible spans, each span in a line', default='ncbi_train_span_text_sample.txt')
    parser.add_argument('--outfile',type=str, help='Path to an output file', default='ncbi_train_synExact_sample.txt')
    args = parser.parse_args()
    umls_file = getattr(args,'umls')
    input_file = getattr(args, 'infile')
    output_file = getattr(args, 'outfile')   
    bmSearch = multiSearch(umls_file)
    bmSearch.search_synonyms(input_file, output_file)
    
    