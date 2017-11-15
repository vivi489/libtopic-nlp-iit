import re, pickle, sys
from collections import deque

def generate_corpora_on_suggests(f_corpora, suggest_keyword_set, out_path):
    f_out = open(out_path, "w")
    def _scan_n_grams(paragraph_words, suggest_keyword_set, n):
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

    #line_count = 0
    for line in f_corpora:
        #if line_count%500==0: print "paragraph", line_count
        #line_count += 1
        paragraph_words = [w for w in re.split("\s+", line) if len(w)>0]
        for n in range(2, 7):
            paragraph_len = len(paragraph_words)
            while True:
                paragraph_words = _scan_n_grams(paragraph_words, suggest_keyword_set, n)
                if paragraph_len == len(paragraph_words): break
                paragraph_len = len(paragraph_words)
        f_out.write(" ".join(paragraph_words))
        f_out.write('\n')
    f_out.close()

# python concatenate_tokens_into_keywords.py syukatsu_token.txt set_valid_suggest_words_syukatsu
# python concatenate_tokens_into_keywords.py jawiki_token.txt set_valid_suggest_words_syukatsu jawiki_token_concat_syukatsu.txt

if __name__ == "__main__":

    f_corpora = open(sys.argv[1], 'r')
    suggest_keyword_set = pickle.load(open(sys.argv[2], 'r'))
    if len(sys.argv)<4:
        f_out_name = "%s_concat.txt"%sys.argv[1][:-4]
    else:
        f_out_name = sys.argv[3]
    print "initialized"
    generate_corpora_on_suggests(f_corpora, suggest_keyword_set, f_out_name)

