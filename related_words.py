import pickle
from gensim.models import Phrases
import gensim
from gensim.models import Word2Vec
import Cython
import re

model = Word2Vec.load("Models/word2vec_50.model")
# bigram = Word2Vec.load("Models/trigram_10.model")

kv = model.wv
global_dict = kv.key_to_index
# with open('global_dict.pickle', 'wb') as f:
#     pickle.dump(global_dict, f)


# kvb = bigram.wv
# global_dict_bigram = kvb.key_to_index
# with open('global_bigram_dict.pickle', 'wb') as f:
#     pickle.dump(global_dict_bigram, f)

# bow = str(list(global_dict.keys()))
# bg_bow = str(list(global_dict_bigram.keys()))


def find_things(pat, s):
    p = fr'(\w*{pat}\w*)'
    return re.findall(p, s)


stem_words = ['inflat', 'pric', 'consum', 'expendi', 'cpi', 'produc', 'econ']
# derived_words = []
derived_words = ['inflation', 'inflationary', 'disinflation', 'deflation', 'hyperinflation', 'price', 'consumer', 'consumption',
                 'consume', 'expenditure', 'cpi', 'produce', 'production', 'producer', 'economic', 'economy', 'macroeconomic']
derived_words = ['inflation']
# for i in stem_words:
#     derived_words.append(find_things(i, bg_bow))
    # print(find_things(i, bg_bow))

related_words = []
failed = []
for i in derived_words:
    # for e in i:
        try:
            # # Trigrams
            # temp = [(e,v) for e, v in kv.most_similar_cosmul(i, topn=47) if re.search(r'\b[a-z]+_[a-z]+_[a-z]+\b', e) and v > .7]
            temp = [(e, v) for e, v in kv.most_similar_cosmul(i, topn=47)]
            related_words += temp[:10]
        except:
            failed.append(i)
            continue

with open('../Databases/Data/Related/related_trigrams_50.pickle', 'wb') as handle:
    pickle.dump(related_words, handle, protocol=pickle.HIGHEST_PROTOCOL)