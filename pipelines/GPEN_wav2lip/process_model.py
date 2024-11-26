import torch
import pprint

def print_model_structure(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    for key in state_dict.keys():
        print(key)

model_path = "pretrained_ffhq.pt"
print_model_structure(model_path)