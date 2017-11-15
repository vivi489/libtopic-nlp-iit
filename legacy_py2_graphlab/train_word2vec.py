# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# python train_word2vec_Chen.py 5 jawiki_token_concat_no_kafunsyo.txt
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.WARNING)
    logger.warning("running %s" % ' '.join(sys.argv))
    
    # check and process input arguments
    if not len(sys.argv) == 3:
        print globals()
        print locals()
        sys.exit(1)
    input = sys.argv[2]
    window = int(sys.argv[1])
    output = os.path.splitext(input)[0].split('_')[-2:]
    output = output[-1] if not output[-2]=="no" else '_'.join(output)
    output = "model_%d_"%window + output + ".model"
    logger.warning("input = %s | output = %s" % (input, output))

    txtData = LineSentence(input)
    model = Word2Vec(txtData, size=256, window=window, min_count=5, workers=multiprocessing.cpu_count())
    
    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model.save(output)
