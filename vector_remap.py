import pickle
import pandas
import multiprocessing

with open("/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/df_count_indices.pickle", "rb") as file1:
    df1 = pickle.load(file1)
    df1 = pandas.DataFrame(df1)

with open("/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/vocab_dict.pickle", "rb") as file2:
    global_dict = pickle.load(file2)


def remap_vector(inverse_local_dict):
    vector = inverse_local_dict[0]
    inverse_local_dict = inverse_local_dict[1]
    for index, v in enumerate(vector):
        vector[index] = global_dict[inverse_local_dict[v]]
    return vector


def refactor_dicts(inverse_local_dict):
    inverse_local_dict = inverse_local_dict[1]
    local_dict = {v: global_dict[v] for key, v in inverse_local_dict.items()}
    return local_dict


def parallelize_remap( inverse_local_dict, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(num_workers) as pool:
        dicts = pool.map(remap_vector, inverse_local_dict)
    return dicts

if __name__ == "__main__":
    df1['remapped_vectors'] = parallelize_remap(list(zip(df1['vectors'], df1['inverse_vocabs'])))

    with open("/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/df_count_indices.pickle", "wb") as file3:
        df1 = pickle.dump(df1, file3)