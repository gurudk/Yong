import torch
import torchtext.datasets as datasets

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()

# train, val, test = datasets.Multi30k(language_pair=("de", "en"))


train = datasets.Multi30k(root='data', split='train', language_pair=('de', 'en'))
val = datasets.Multi30k(root='data', split='valid', language_pair=('de', 'en'))
test = datasets.Multi30k(root='data', split='test', language_pair=('de', 'en'))
