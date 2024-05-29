from argparse import Namespace
import time
import os
import sys
import clip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import subprocess

import torchvision.transforms as transforms

import os

CODE_DIR = 'encoder4editing'
os.chdir(f'./{CODE_DIR}')

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder

os.chdir("/home/ayavasileva/StyleCLIP/global_torch")
import clip
import dnnlib
import legacy

sys.path.append('/home/ayavasileva/StyleCLIP/global_torch')
sys.path.append('/home/ayavasileva/StyleCLIP/global_torch/training')

from manipulate import Manipulator

network_pkl='/home/ayavasileva/stylegan2-ffhq-config-f.pkl'
device = torch.device('cuda')

M=Manipulator()
G=M.LoadModel(network_pkl,device)

with dnnlib.util.open_url(network_pkl) as fp:
        D = legacy.load_network_pkl(fp)['D'].requires_grad_(False).to(device) # type: ignore
        
def generate_latents(G, D, batch_size=20, n_batches=1000, output_folder="/home/ayavasileva/data/generated_im", w_file_name="/home/ayavasileva/data/ws_gen"):
    all_ws = []
    all_imgs = []
    all_probs = []
    total_im = 0

    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for batch in range(n_batches):
            z = torch.randn(batch_size, 512).to("cuda")
            ws = G.mapping(z.to("cuda:0"), c=None)
            images = G.synthesis.forward(ws)
            scores = D(images, c=None)
            scores = np.array(torch.sigmoid(scores).detach().cpu().flatten())
            mask = (scores > 0.61) & (scores < 0.77)
            all_probs.extend(scores[mask])
            all_imgs.extend(images[mask])
            all_ws.append(ws[mask].detach().cpu().numpy())
            for n_image, image in enumerate(images[mask]):
                    new_image = tensor2im(images[n_image])
                    new_image.save(f"{output_folder}/{total_im + n_image}.jpg")  
            total_im += images[mask].shape[0]
        np.save(w_file_name, np.concatenate(all_ws, axis=0))
