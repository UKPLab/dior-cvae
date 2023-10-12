import json

result_path = './dailydialog/'
path = './dailydialog/processed/'
test_data =[]
valid_data = []
train_data = []
with open(path + 'test.src') as f:
    test_src = f.readlines()
with open(path + 'test.tgt') as f:
    test_tgt = f.readlines()
with open(path + 'valid.src') as f:
    valid_src = f.readlines()
with open(path + 'valid.tgt') as f:
    valid_tgt = f.readlines()
with open(path + 'train.src') as f:
    train_src = f.readlines()
with open(path + 'train.tgt') as f:
    train_tgt = f.readlines()


for (src, tgt) in zip(test_src, test_tgt):
    temp = {}
    temp['source'] = src.strip()
    temp['target'] = tgt.strip()
    test_data.append(temp)

for (src, tgt) in zip(valid_src, valid_tgt):
    temp = {}
    temp['source'] = src.strip()
    temp['target'] = tgt.strip()
    valid_data.append(temp)

for (src, tgt) in zip(train_src, train_tgt):
    temp = {}
    temp['source'] = src.strip()
    temp['target'] = tgt.strip()
    train_data.append(temp)

with open(result_path + 'test.txt', 'w') as f:
    for d in test_data:
        f.write(json.dumps(d) + '\n')

with open(result_path + 'valid.txt', 'w') as f:
    for d in valid_data:
        f.write(json.dumps(d) + '\n')

with open(result_path + 'train.txt', 'w') as f:
    for d in train_data:
        f.write(json.dumps(d) + '\n')