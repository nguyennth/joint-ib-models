This is the source code for the paper entitled "Span-based Named Entity Recognition by Generating and Compressing Information". 

It is noted that most of the code for processing data was forked from the DeepEventMine repo: https://github.com/aistairc/DeepEventMine

# 0. Setup environment
```bash
python3 -m venv joint-ib-env
source joint-ib-env/bin/activate
cd $HOME/joint-ib-models
pip install -r requirements.txt
```

# 1. Prepare data 
## 1.1 Corpus
- It is noted that our model receives tokenised data in brat format (https://brat.nlplab.org/standoff.html) in both training and testing stages as shown in data/ncbi

## 1.2 Data for synonym generation
- Firstly, we have to preprocess UMLS to have a file in which each line represent a term in UMLS with its CUIs.
- We have to print all enumerated spans from the train, dev, test sets
- Run loader/exact_matching.py to extract synonyms for each span
- Here we give synonyms for the NCBI corpus as an example

## 1.3 Download wordvectors for the decoder
```bash
mkdir data/wordvec
cd data/wordvec
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hqd14uyS8_YbdoBckbWWo7dpiK3iAlZh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hqd14uyS8_YbdoBckbWWo7dpiK3iAlZh" -O wiki_biowordvec.200d.tar.gz && rm -rf /tmp/cookies.txt
tar -zxvf wiki_biowordvec.200d.tar.gz wiki_biowordvec.200d.txt
```

It is noted that these word vectors were post-processed from this: http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin

# 2. Configurations
- All configs should be specified in a yaml file as shown in the folder experiments/ncbi. In the ncbi-train.yaml file, we provide parameters of the joint model producing the best performance on NCBI.
- Some configurations can be overwritten by values from a command line while training the model. However, all when testing the model, configurations are loaded from the trained parameters file.

# 3. Train the model
```bash
python3 src/train.py --yaml experiments/ncbi/ncbi-train.yaml
```
# 4. Evaluate the model
```bash
python3 src/predict.py --yaml experiments/ncbi/ncbi-test.yaml
``` 
Both F1-score for NER and BLEU-2 score for span generation will be shown.

# Citation
If you use this repository, please cite this paper:

Nhung TH Nguyen, Makoto Miwa, and Sophia Ananiadou. Span-based Named Entity Recognition by Generating and Compressing Information. In proceedings of EACL 2023.