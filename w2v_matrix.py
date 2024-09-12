import gc
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
from collections import Counter

# with open('../Databases/Data/Globals/words_15k.pkl', 'rb') as handle:
#     data = pickle.load(handle)

related_words = [('deflation', 0.8514788150787354), ('inflationary', 0.8282365202903748), ('policymaker', 0.8109442591667175), ('fed', 0.800275981426239), ('growth', 0.800038754940033), ('stubbornly', 0.7975570559501648), ('transitory', 0.7949220538139343), ('slowing', 0.7914058566093445), ('weakening', 0.7870364785194397), ('unemployment', 0.7832043170928955), ('economy', 0.7793191075325012), ('contraction', 0.7778199911117554), ('persistently', 0.7758837342262268), ('price', 0.7746670842170715), ('downward', 0.7713813185691833), ('rate', 0.7686709761619568), ('eurozone', 0.7675325870513916), ('slowdown', 0.7667698860168457), ('modestly', 0.7650092840194702), ('joblessness', 0.763563871383667), ('recession', 0.7625768184661865), ('deficit', 0.7611256241798401), ('deflationary', 0.7600934505462646), ('borrowing', 0.7573513984680176), ('volatility', 0.7570568323135376), ('pace', 0.7561020255088806), ('g.d.p', 0.7548074722290039), ('c.p.i', 0.7510446906089783), ('output', 0.7493780255317688), ('sluggish', 0.7460706830024719), ('upward', 0.7441329956054688), ('upturn', 0.7366918921470642), ('productivity', 0.7364533543586731), ('trough', 0.7359928488731384), ('economist', 0.7353026270866394), ('seasonally', 0.7345349192619324), ('stagnation', 0.7322917580604553), ('artificially', 0.7300978899002075), ('tightening', 0.7297743558883667), ('macroeconomic', 0.7297592163085938), ('bernanke', 0.7295627593994141), ('decelerate', 0.7281503081321716), ('downturn', 0.7274260520935059), ('f.o.m.c', 0.7269579172134399), ('anemic', 0.7264103889465332), ('birthrate', 0.7260218858718872), ('nominal', 0.7253251075744629)]
# for e in data:
#     # for e in i:
#         if '_' in e[0] and e[0] not in related_words:
#             related_words.append(e[0])
# related_words = data
related_words = [e for e, v in related_words]
related_words = ['republicans', 'cut', 'sex', 'shelter', 'slave', 'condemn', 'lend', 'japanese', 'medicine', 'label', 'eliminate']

with open('../Databases/Data/df_count.pickle', 'rb') as f:
    df_count = pickle.load(f)

def count_words_in_text(tokens):
    row_vec = []
    for i in related_words:
        row_vec.append(tokens[i])
    return row_vec


# Function to count words in multiple texts using parallel processing
def parallel_count_words(counters, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(count_words_in_text, counters)
    return matrix


if __name__ == '__main__':
    np_matrix = parallel_count_words(df_count['word_counts'].tolist())

    dates = df_count['date'].tolist()
    for index, i in enumerate(dates):
        np_matrix[index].insert(0, i)

    np_matrix = np.array(np_matrix)
    df2 = pd.DataFrame(np_matrix, columns=['date', *related_words])

    # memory management
    del df_count
    gc.collect()

    with open('../Databases/Data/Matrices/tree.pickle', 'wb') as handle:
        pickle.dump(df2, handle)
