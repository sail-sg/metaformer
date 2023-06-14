import torch
from collections import OrderedDict
import argparse
import json



parser = argparse.ArgumentParser()

parser.add_argument('checkpoint', help='21k pretrained checkpoint')
parser.add_argument('--in21k_1k_map', default='in21k_to_1k_class_map.json', help='in21k to 1k class map')
parser.add_argument('--output_ckpt', default='processed.pth', help='path to save processed checkpoint')

args = parser.parse_args()

state_dict = torch.load(args.checkpoint)

with open(args.in21k_1k_map, 'r') as f:
    class_map = json.load(f)


for k in state_dict.keys():
    if state_dict[k].shape[0] == 21841:
        new_shape = list(state_dict[k].shape)
        new_shape[0] = 1000
        new_weights = torch.zeros(new_shape, dtype=state_dict[k].dtype)
        new_weights[class_map['in1k_idx']] = state_dict[k][class_map['in21k_idx']]
        new_weights[850] = state_dict[k].mean(0) # class 850 is missing, just use the mean to init it
        state_dict[k] = new_weights

torch.save(state_dict, args.output_ckpt)