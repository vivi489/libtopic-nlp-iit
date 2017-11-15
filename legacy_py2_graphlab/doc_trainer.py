# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import TaggedLineDocument

import graphlab, gensim, sys
import logging

class DocBatchGenerator(object):
    def __init__(self, f):
        self.f = f
        self.line_index = -1
    
    def __iter__(self):
        for line in self.f:
            self.line_index += 1
            yield TaggedDocument(line, [self.line_index])


if not len(sys.argv)==3:
    exit(0)

logging.basicConfig(filename='d2v_training_%s.log'%sys.argv[2], format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Doc2Vec(dm = 1, window=5, alpha=0.025, size=300, min_alpha=0.025, min_count=1, sample=1e-6, workers=8)
model.build_vocab(TaggedLineDocument(open(sys.argv[1], 'r')))

# start training
num_epochs = 600

for e in xrange(num_epochs):
    if e%20==0:
        logging.info('%s: now training epoch %d'%(sys.argv[2], e))
    model.train(DocBatchGenerator(open(sys.argv[1], 'r')), total_examples=model.corpus_count, epochs=1)
    #model.alpha -= (0.025 - 0.0001) / float(num_epochs-1)  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


model.save("doc2vec_model_%s"%sys.argv[2])

