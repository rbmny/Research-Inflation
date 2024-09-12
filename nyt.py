import pickle
import pandas

# open a file, where you stored the pickled data
file = open('../Databases/NYT_database.pickle', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print('Showing the pickled data:')

cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    cnt += 1

print(data['date'])

vocab = []

ncnt = 0
for item in data['related_tags'].astype(str):
    if item != "nan":
        break
    ncnt += 1

wcnt = 0
for item in data['related_tags'].astype(str):
    words = item.split(', ')
    for word in words:
        if word not in vocab and not any(c.isdigit() for c in word):
            vocab.append(word)
            wcnt += 1

print(vocab, wcnt, ncnt)