import torch
from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import Dataset


def get_classes(all_claz, main_cls):
    clses = [cls for cls in main_cls.__subclasses__()]
    for clz in clses:
        all_claz.append(clz)
        sub_clazz = [sub_clz for sub_clz in clz.__subclasses__()]
        if sub_clazz:
            get_classes(all_claz, clz)


print([cls.__name__ for cls in Dataset.__subclasses__()])
all_clazz = []

get_classes(all_clazz, Dataset)
print(all_clazz)
