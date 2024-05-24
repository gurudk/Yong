from datasets import load_dataset

data_path = "/home/wolf/mygit/wmt14/fr-en/"
dataset = load_dataset("parquet", data_files={'train': data_path + 'train-00000-of-00030.parquet'},
                       split='train[10:20]')

for example in dataset:
    print(example["translation"]["en"])
    print(example["translation"]["fr"])
