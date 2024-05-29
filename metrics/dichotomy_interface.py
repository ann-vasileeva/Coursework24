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
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.


def make_folder_with_attribute(attr_name: str, col_name: str, annotation, create_imgs=True):
    img_dir = "/home/ayavasileva/data/inverted_celeba"

    if not col_name in annotation.columns:
        print("No such column")
        return

    files_with_attribute = annotation[annotation[col_name] == 1]["path"]
    files_without_attribute = annotation[annotation[col_name] != 1]["path"]
    print(len(files_with_attribute))
    print(len(files_without_attribute))
    without_attribute_idx = annotation[annotation[col_name] != 1].index
    print(len(without_attribute_idx))
    new_dir = f'/home/ayavasileva/data/files_with_{attr_name}3000'
    new_without_dir = f'/home/ayavasileva/data/files_without_{attr_name}3000'

    if create_imgs:

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if not os.path.exists(new_without_dir):
            os.mkdir(new_without_dir)

        for im_path in files_with_attribute:
            source_file_path = os.path.join(img_dir, im_path)
            target_file_path = os.path.join(new_dir, im_path)
            
            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, target_file_path)

        for im_path in files_without_attribute:
            source_file_path = os.path.join(img_dir, im_path)
            target_file_path = os.path.join(new_without_dir, im_path)

            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, target_file_path)

    return without_attribute_idx
                        
                        
def apply_interface_batches(net, latents, direction,factor,latent_idx, create_photos=True, output_folder="interface_generated"):
    b_sz = 25
    n_batches = len(latents) // b_sz
    if len(latents) % b_sz != 0:
        n_batches += 1

    if not os.path.exists(f"/home/ayavasileva/data/{output_folder}_strength{np.abs(factor)}"):
        os.mkdir(f"/home/ayavasileva/data/{output_folder}_strength{np.abs(factor)}")

    for batch in range(n_batches):
       edit_latents = latents[batch * b_sz: (batch+1)*b_sz].cuda() + factor * direction
       if create_photos:
            with torch.no_grad():
                images, _ = net([edit_latents.to("cuda")], randomize_noise=False, input_is_latent=True)
                for n_image, image in enumerate(images):
                    new_image = tensor2im(image)
                    new_image.save(f"/home/ayavasileva/data/{output_folder}_strength{np.abs(factor)}/{latent_idx[n_image + batch * b_sz]}.jpg")                        
                        
                                               
def binary_search_for_s(net, latents_to_edit, ganspace_pca, strengths, inds_without, direction, method_name="ganspace", direction_name="smile", celeba_attr="Smiling"): #unimodal function
    # values = []
    # f_val = []
    known_fid_scores = {}
    left = 0
    right = len(strengths) - 1
    mid = left + (right - left) // 2

    #count mid value
    apply_interface_batches(net.decoder, latents_to_edit, interfacegan_direction,
                            strengths[mid], inds_without,output_folder=f"{method_name}_generated_{direction_name}")
                            
    os.chdir("/home/ayavasileva/sfe-main/launch")

    cur_edit_path = f"/home/ayavasileva/data/{method_name}_generated_{direction_name}_strength{np.abs(strengths[mid])}"

    result = subprocess.run(['python', 'fid_calculation.py', f'--orig_path=/home/ayavasileva/data/files_with_{direction_name}3000',
                f'--synt_path={cur_edit_path}',
                            f'--attr_name={celeba_attr}',  '--celeba_attr_table_pth=/home/ayavasileva/3000.txt'],
                            capture_output=True, text=True)

    fid_score_mid = float(result.stdout.split('\n')[-2].split()[-1])
    known_fid_scores[strengths[mid]] = fid_score_mid
    subprocess.run(['rm', '-rf', cur_edit_path])

    # values.append(strengths[mid])
    # f_val.append(fid_score_mid)

    while left < right:
      if right - left == 1:
      	best_score = min(known_fid_scores[strengths[left]], known_fid_scores[strengths[right]])
      	print("best score is: ", best_score)
        
      y = left + (mid - left) // 2
      print("Left, right, mid:")
      print(left, mid, right)
      print("Strengths: ")
      print(strengths[left], strengths[mid], strengths[right])

      #count y value
      if strengths[y] not in known_fid_scores:
        apply_interface_batches(net.decoder, latents_to_edit, interfacegan_direction,
                            strengths[y], inds_without,output_folder=f"{method_name}_generated_{direction_name}")

        cur_edit_path = f"/home/ayavasileva/data/{method_name}_generated_{direction_name}_strength{np.abs(strengths[y])}"

        result = subprocess.run(['python', 'fid_calculation.py', f'--orig_path=/home/ayavasileva/data/files_with_{direction_name}3000',
                  f'--synt_path={cur_edit_path}', f'--attr_name={celeba_attr}',  '--celeba_attr_table_pth=/home/ayavasileva/3000.txt'],
                              capture_output=True, text=True)

        fid_score_y = float(result.stdout.split('\n')[-2].split()[-1])
        known_fid_scores[strengths[y]] = fid_score_y

        subprocess.run(['rm', '-rf', cur_edit_path])
      else:
        fid_score_y = known_fid_scores[strengths[y]]

      print(f"fid_score={fid_score_y} for strength={strengths[y]}")
      print(f"fid_score={fid_score_mid} for strength={strengths[mid]}")

      # values.append(strengths[y])
      # f_val.append(fid_score_y)

      if fid_score_y <= fid_score_mid:
        right = mid
        mid = y
        fid_score_mid = fid_score_y

      else:
        z = mid + (right - mid)//2

        #count z value
        if strengths[z] not in known_fid_scores:
          apply_interface_batches(net.decoder, latents_to_edit, interfacegan_direction,
                            strengths[z], inds_without,output_folder=f"{method_name}_generated_{direction_name}")

          cur_edit_path = f"/home/ayavasileva/data/{method_name}_generated_{direction_name}_strength{np.abs(strengths[z])}"

          result = subprocess.run(['python', 'fid_calculation.py', f'--orig_path=/home/ayavasileva/data/files_with_{direction_name}3000',
                  f'--synt_path={cur_edit_path}',f'--attr_name={celeba_attr}',  '--celeba_attr_table_pth=/home/ayavasileva/3000.txt'],
                              capture_output=True, text=True)

          fid_score_z = float(result.stdout.split('\n')[-2].split()[-1])
          subprocess.run(['rm', '-rf', cur_edit_path])
        else:
          fid_score_z = known_fid_scores[strengths[z]]

        print(f"fid_score={fid_score_z} for strength={strengths[z]}")

        # values.append(strengths[z])
        # f_val.append(fid_score_z)

        if fid_score_mid <= fid_score_z:
          left = y
          right = z
        else:
          left = mid
          mid = z
          fid_score_mid = fid_score_z

    print(f"best_score is: {fid_score_mid} and best strength is: {strengths[mid]}")
    #return fid_score_mid, mid,values,f_val
    
    
experiment_type = 'ffhq_encode'

EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "/home/ayavasileva/encoder4editing/pretrained_models/e4e_ffhq_encode.pt",
    }
}
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
img_transforms = EXPERIMENT_ARGS['transform']

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
#update the training options
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')


os.chdir("..")
annotation = pd.read_csv("/home/ayavasileva/3000.txt", delimiter=" ",
                         skiprows=[0])
annotation.reset_index(inplace=True)
annotation.drop(columns=["level_1"], inplace=True)
annotation.rename(columns={'level_0': 'path'}, inplace=True)


from editings import interface

interfacegan_directions = {
    'ffhq_encode': {
        'age': 'editings/interfacegan_directions/age.pt',
        'smile': 'editings/interfacegan_directions/smile.pt',
        'pose': 'editings/interfacegan_directions/pose.pt'
    }
}

interfacegan_direction = torch.load(interfacegan_directions[experiment_type]["smile"]).cuda()

inds_without = make_folder_with_attribute(attr_name="smile", col_name="Smiling", annotation=annotation)
#np.save(inds_without, "inds_without_smile.npy")
print(len(os.listdir("/home/ayavasileva/data/files_without_smile3000")))
ws = np.load("/home/ayavasileva/data/all_ws.npy")
print(ws.shape)
latents_to_edit = torch.tensor(ws[inds_without])
#torch.save(latents_to_edit, "latents_to_edit_smile.pt")
print(latents_to_edit.shape)

strengths = strengths = np.linspace(0, 4, 20)

binary_search_for_s(net, latents_to_edit, interfacegan_direction, strengths, inds_without, method_name="interface", direction_name="smile",celeba_attr="Smiling")
                                                                                                     
