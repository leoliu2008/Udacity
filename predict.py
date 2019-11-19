import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torchvision import datasets, transforms, models

# from collections import OrderDict

import pandas as pd
import numpy as np

from PIL import Image
import argparse
import util

import json

arg=argparse.ArgumentParser(description='predict.py')

arg.add_argument('data_dir', action='store', default='flowers', type = str)

# arg.add_argument('image_path', action='store', default='flowers/test/10/image_07090.jpg', nargs='?', type = str)

arg.add_argument('load_dir', action='store', default='checkpoint.pth', type=str)

arg.add_argument('--gpu', dest='gpu',action='store', default='gpu')
arg.add_argument('--top_k',dest='top_k', action='store',default=5, type=int)
arg.add_argument('--category_name',dest='category_name',default='cat_to_name.json', action='store',type=str)

pa=arg.parse_args()

image_path= pa.data_dir + '/test' + '/10/' + 'image_07090.jpg'


# print(image_path)

def main():

    model=util.load_checkpoint(pa.load_dir)


    with open(pa.category_name, 'r') as f:
        cat_to_name = json.load(f)
    
#     print(cat_to_name.get)
    
#     print(len(cat_to_name))
    
    top_p_list, top_flowers = util.predict(image_path, pa.load_dir, cat_to_name, pa.top_k)
#     labels = [cat_to_name[str(index+1)] for index in np.naray(top_p_list[1][0])
            
#     return print(top_p_list[0])       
    i=0
    while i <pa.top_k: 
        print('{} with a probability of {}'.format(top_flowers[i], top_p_list[i]))
        i +=1            
    
if __name__ =='__main__':
    main()