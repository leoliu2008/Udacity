import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torchvision import datasets, transforms, models

import pandas as pd
import numpy as np
from PIL import Image


import argparse
import util
from workspace_utils import active_session


arg=argparse.ArgumentParser(description='train.py')

arg.add_argument('data_dir', action='store', default='flowers')
arg.add_argument('--gpu', dest='gpu',action='store', default='gpu')
arg.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')

arg.add_argument('--learning_rate',dest='learning_rate', action='store',default=0.005,type=float)
arg.add_argument('--drop_out',dest='drop_out',action='store',default=0.15,type=float)
arg.add_argument('--epochs',dest='epochs',action='store',default=1,type=int)
arg.add_argument('--model_type',dest='model_type',action='store',default='vgg16',type=str)
arg.add_argument('--hidden_layer_1',dest='hidden_layer_1',action='store',default=1024,type=int)
arg.add_argument('--hidden_layer_2',dest='hidden_layer_2',action='store',default=512,type=int)
arg.add_argument('--output_units',dest='output_units',action='store',default=102,type=int)

pa=arg.parse_args()






def main():
    trainloaders, validloaders, testloaders, train_data = util.transform_load_image(pa.data_dir)
    model, criterion, optimizer = util.construct_newwork(pa.model_type, pa.drop_out, pa.hidden_layer_1, pa.hidden_layer_2, pa.output_units, pa.learning_rate)

    with active_session():
        util.test_network(model, criterion, optimizer,trainloaders, validloaders, pa.epochs, print_every=40, steps=0)
        util.save_checkpoint(model, train_data, optimizer, pa.save_dir, pa.epochs)

    print('Training completed')

if __name__ =='__main__':
    main()