import pickle

with open('../Databases/Data/Globals/text_list.pickle', 'rb') as f:
        tokenized_docs = pickle.load(f)
        print(len(tokenized_docs))