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
                        
                        
def apply_stylespace_batches(net, latents, strength, latent_idx, create_photos=True, output_folder="stylespace_generated"):

    b_sz = 10
    n_batches = len(latents) // b_sz
    if len(latents) % b_sz != 0:
        n_batches += 1
    
    img_index = 0
   # img_indexs=[i for i in range(b_sz)]

    M.alpha=[strength]
    lindex, cindex = 9,6  #wavy_hair

    n_images = len(latents) 

    if not os.path.exists(f"/home/ayavasileva/data/{output_folder}_strength{np.abs(strength)}"):
        os.mkdir(f"/home/ayavasileva/data/{output_folder}_strength{np.abs(strength)}")

    for ind in range(n_batches):
       edit_latents = latents[ind*b_sz: (ind+1)*b_sz].cuda()
      #  torch.save(edit_latents, "/content/latents.pt")
      #  edit_latents = torch.load("/content/latents.pt")
       dlatents_loaded=M.G.synthesis.W2S(edit_latents)
       batch_len = edit_latents.shape[0]
       del edit_latents
       torch.cuda.empty_cache() 
       dlatents_loaded=M.S2List(dlatents_loaded)
       img_indexs = [i for i in range(batch_len)]
       dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]
       M.num_images= batch_len
       M.manipulate_layers=[lindex]
       with torch.no_grad():
        _,images=M.EditOneC(cindex, dlatent_tmp)
        for n_im , image in enumerate(images):
          image = Image.fromarray(image[0])
          image.save(f"/home/ayavasileva/data/{output_folder}_strength{np.abs(strength)}/{latent_idx[ind*b_sz + n_im]}.jpg")
        del images
        torch.cuda.empty_cache()                        
                        
                                               
def binary_search_for_s(net, latents_to_edit, strengths, inds_without, method_name="ganspace", direction_name="smile", celeba_attr="Smiling"): #unimodal function
    # values = []
    # f_val = []
    known_fid_scores = {}
    left = 0
    right = len(strengths) - 1
    mid = left + (right - left) // 2

    #count mid value
    apply_stylespace_batches(net.decoder, latents_to_edit, strengths[mid], inds_without, create_photos=True, output_folder=f"stylespace_generated_{direction_name}")
                            
    os.chdir("/home/ayavasileva/sfe-main/launch")

    cur_edit_path = f"/home/ayavasileva/data/{method_name}_generated_{direction_name}_strength{np.abs(strengths[mid])}"

    result = subprocess.run(['python', 'fid_calculation.py', f'--orig_path=/home/ayavasileva/data/files_with_{direction_name}3000',
                f'--synt_path={cur_edit_path}',
                            f'--attr_name={celeba_attr}',  '--celeba_attr_table_pth=/home/ayavasileva/3000.txt'],
                            capture_output=True, text=True)

    print(result.stdout)
#    print(result.stderr)
    fid_score_mid = float(result.stdout.split('\n')[-2].split()[-1])
    known_fid_scores[strengths[mid]] = fid_score_mid
    subprocess.run(['rm', '-rf', cur_edit_path])

    # values.append(strengths[mid])
    # f_val.append(fid_score_mid)

    while left < right:
      if right - left == 1:
      	best_score = min(known_fid_scores[strengths[left]], known_fid_scores[strengths[right]])
      	print("best score is: ", best_score)
      	return None
        
      y = left + (mid - left) // 2
      print("Left, right, mid:")
      print(left, mid, right)
      print("Strengths: ")
      print(strengths[left], strengths[mid], strengths[right])

      #count y value
      if strengths[y] not in known_fid_scores:
        apply_stylespace_batches(net.decoder, latents_to_edit, strengths[y], inds_without, create_photos=True, output_folder=f"stylespace_generated_{direction_name}")


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
          apply_stylespace_batches(net.decoder, latents_to_edit, strengths[z], inds_without, create_photos=True, output_folder=f"stylespace_generated_{direction_name}")
         
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


os.chdir("/home/ayavasileva/StyleCLIP/global_torch")
import clip

sys.path.append('/home/ayavasileva/StyleCLIP/global_torch')
sys.path.append('/home/ayavasileva/StyleCLIP/global_torch/training')

from manipulate import Manipulator
from StyleCLIP import GetDt,GetBoundary

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/ayavasileva/ViT-B-32.pt", device=device,jit=False)

network_pkl='/home/ayavasileva/stylegan2-ffhq-config-f.pkl'
device = torch.device('cuda')
M=Manipulator()
M.device=device
G=M.LoadModel(network_pkl,device)
M.G=G
M.SetGParameters()
num_img=100_000
M.GenerateS(num_img=num_img)
M.GetCodeMS()

file_path='./npy/ffhq/'
fs3=np.load(file_path+'fs3.npy')

inds_without = make_folder_with_attribute(attr_name="male", col_name="Male", annotation=annotation)
#np.save(inds_without, "inds_without_smile.npy")
print(len(os.listdir("/home/ayavasileva/data/files_without_male3000")))
ws = np.load("/home/ayavasileva/data/all_ws.npy")
print(ws.shape)
latents_to_edit = torch.tensor(ws[inds_without])
#torch.save(latents_to_edit, "latents_to_edit_smile.pt")
print(latents_to_edit.shape)

strengths = strengths = np.linspace(-8,25,35)

binary_search_for_s(net, latents_to_edit, strengths, inds_without, method_name="stylespace", direction_name="male",celeba_attr="Male")
                                                                                     
