from datasets import load_dataset
#
# dataset = load_dataset("parquet", data_files={'train': 'data/wmt14/train-00000-of-00030.parquet'})
import datasets

ds = datasets.load_dataset("wmt/wmt14", 'fr-en')
ds.save_to_disk("data/wmt14/fr-en")
