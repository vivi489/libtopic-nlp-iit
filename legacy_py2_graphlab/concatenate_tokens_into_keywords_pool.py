from multiprocessing import Process, Pool
from collections import deque
import re, pickle, sys, random, time, copy


BATCH_SIZE = 256
POOL_SIZE = 16

def generate_corpora_on_suggests((paragraph, suggest_keyword_set)):
    def _scan_n_grams(paragraph_words, n):
        retVal = []
        q = deque(maxlen=n)
        for w in paragraph_words:
            if len(q)<n:
                q.append(w)
                continue;
            cur_gram = "".join(q)
            if cur_gram in suggest_keyword_set:
                retVal.append(cur_gram)
                q.clear()
                q.append(w)
            else:
                retVal.append(q[0])
                q.append(w)
        while len(q)>0: retVal.append(q.popleft())
        return retVal
    #if line_count%5000==0: print "paragraph", line_count
    #line_count += 1
    paragraph_words = [w for w in re.split("\s+", paragraph) if len(w)>0]
    for n in range(2, 7):
        paragraph_len = len(paragraph_words)
        while True:
            paragraph_words = _scan_n_grams(paragraph_words, n)
            if paragraph_len == len(paragraph_words): break
            paragraph_len = len(paragraph_words)
    return " ".join(paragraph_words)


def repeat(x, times=None):
    remain = times
    while remain > 0:
        remain -= 1
        yield x

# python concatenate_tokens_into_keywords.py jawiki_token.txt set_valid_suggest_words_syukatsu jawiki_token_concat_no_syukatsu.txt
if __name__ == '__main__':
    
    if not len(sys.argv)==4:
        exit(0)
    f_corpora = open(sys.argv[1], 'r')
    #suggest_keyword_set = pickle.load(open(sys.argv[2], 'rb'), encoding="utf-8") #python 3 compat.
    suggest_keyword_set = pickle.load(open(sys.argv[2], 'rb'))

    f_out_name = sys.argv[3]

    eof = False
    while True:
        line_batch = []
        while len(line_batch) < BATCH_SIZE:
            line = f_corpora.readline()
            if len(line) == 0:
                eof = True
                break
            line_batch.append(line.strip())
        if len(line_batch) == 0: break
        p = Pool(POOL_SIZE)
        
        ans = p.imap(generate_corpora_on_suggests, zip(line_batch, repeat(suggest_keyword_set, times=len(line_batch))))
        fout = open(f_out_name, 'a')
        for t in ans:
            fout.write(t+'\n')
        fout.close()
        if eof: break
            
        p.close()
        p.join()
    f_corpora.close()

