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


class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("/home/ayavasileva/ViT-B-32.pt", device="cuda")
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.reg = 0
        self.loss = torch.nn.MSELoss()
        self.transform = transforms.Normalize((-1, -1, -1), (2, 2, 2))
        self.transform_clip = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def forward(self, image1, image2, frozen_features, target_text):

        image2 = torch.clamp(self.transform(image2), 0.0, 1.0)
        image2 = self.transform_clip(self.avg_pool(image2))
        image1 = torch.clamp(self.transform(image1), 0.0, 1.0)
        image1 = self.transform_clip(self.avg_pool(image1))

        logits_per_image_init, logits_per_text_init = self.model(image1, frozen_features)
        logits_per_image, logits_per_text = self.model(image2, frozen_features)


        lambda_save = 0.003


        self.reg = lambda_save * self.loss(logits_per_image_init[0], logits_per_image[0]) 
        
        self.sim = (logits_per_image_init[0] - logits_per_image[0]) / 100

        similarity = 1 -  self.model(image2, target_text)[0] / 100 + self.reg  

        return similarity

class IDLoss(torch.nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("/home/ayavasileva/encoder4editing/pretrained_models/model_ir_se50.pth"))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count
       
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp       
       
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
                        
                        
def apply_feature_optimization(net, latents, direction, direction_name="smile", create_photos=True, output_folder="feature_generated"):
    if not os.path.exists(f"{output_folder}_{direction_name}"):
        os.mkdir(f"{output_folder}_{direction_name}")
    frozen_texts = ["face with a big nose", "face with earrings", "female face", "face with beard", "background", "face with lipstick", "face with hair"]
    text_inputs =  ["face", "A smiling person"]
    frozen_features = torch.cat([clip.tokenize(frozen_texts)]).cuda()
    target_text = clip.tokenize(text_inputs[1]).cuda()
    for latent_code_init in latents:
        initial_s = get_stylespace_from_w(latent_code_init, net.decoder)
        with torch.no_grad():
            img_orig, feat = net.decoder(initial_s, is_stylespace=True, randomize_noise=False, return_features=True)
        latent = latent_code_init.detach().clone()  #w case
        latent.requires_grad = True
        steps = 40
        start_lr = 0.1
        lambda_reg = 0.005 #w
        for i in tqdm(range(steps)):
            t = (i) / (steps)
            lr = get_lr(t, start_lr)
            optimizer.param_groups[0]["lr"] = lr
            
            img_gen, _ = net.decoder([latent], input_is_latent=True, randomize_noise=False)

            c_loss = clip_loss(img_orig, img_gen,frozen_features, target_text)
            i_loss = id_loss(img_gen, img_orig)[0]
            lambda_l2 = 0.008
            l2_loss = ((latent_code_init - latent) ** 2).sum()
            
            loss = c_loss + lambda_l2 * l2_loss + lambda_reg * i_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_feature_maps9 = final_feature_maps[9].detach()
        final_feature_maps9.requires_grad = False

        feat_8 = feat[9][mask].detach().clone()
        feat_8.requires_grad = True
        frozen_feat8 = feat[9].detach().clone()
        optimizer = optim.Adam([feat_8], lr=0.01)

        steps = 200
        start_lr = 0.1 #0.1
        lambda_reg = 0.01 #feat

        for i in tqdm(range(steps)):
            t = (i) / (steps)
            lr = get_lr(t, start_lr)
            optimizer.param_groups[0]["lr"] = lr
            
            frozen_feat8[mask] = feat_8

            img_gen, _ = net.decoder(latent_s, is_stylespace=True, randomize_noise=False, inserted_feature=frozen_feat8)
            c_loss = clip_loss(img_orig, img_gen,frozen_features, target_text)

            i_loss = id_loss(img_gen, img_orig)[0]
            lambda_l2  = 0.1
            l2_loss = torch.norm((frozen_feat8 - final_feature_maps[9]), p=2)
            
            loss = c_loss + lambda_l2 * l2_loss + lambda_reg * i_loss 

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
    new_image = tensor2im(img_gen)
    new_image.save(f"{output_folder}_{direction_name}/{n_image:06d}}.png")
                     
    
    
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

os.chdir("/home/ayavasileva/StyleCLIP/models/facial_recognition")
from model_irse import Backbone

os.chdir("..")
annotation = pd.read_csv("/home/ayavasileva/3000.txt", delimiter=" ",
                         skiprows=[0])
annotation.reset_index(inplace=True)
annotation.drop(columns=["level_1"], inplace=True)
annotation.rename(columns={'level_0': 'path'}, inplace=True)


os.chdir("/home/ayavasileva/StyleCLIP/global_torch")
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/ayavasileva/ViT-B-32.pt", device=device,jit=False)

network_pkl='/home/ayavasileva/stylegan2-ffhq-config-f.pkl'
device = torch.device('cuda')


file_path='./npy/ffhq/'
fs3=np.load(file_path+'fs3.npy')

inds_without = make_folder_with_attribute(attr_name="smile", col_name="Smiling", annotation=annotation)
#inds_without = make_folder_with_attribute(attr_name="male", col_name="Male", annotation=annotation)
#inds_without = make_folder_with_attribute(attr_name="gray_hair", col_name="Gray_Hair", annotation=annotation)
#inds_without = make_folder_with_attribute(attr_name="earrings", col_name="Wearing_Earrings", annotation=annotation)
#inds_without = make_folder_with_attribute(attr_name="nose", col_name="Big_Nose", annotation=annotation)
#inds_without = make_folder_with_attribute(attr_name="glasses", col_name="Eyeglasses", annotation=annotation)
#np.save(inds_without, "inds_without_smile.npy")
print(len(os.listdir("/home/ayavasileva/data/files_without_smile3000")))
ws = np.load("/home/ayavasileva/data/all_ws.npy")
print(ws.shape)
latents_to_edit = torch.tensor(ws[inds_without])
#torch.save(latents_to_edit, "latents_to_edit_smile.pt")
print(latents_to_edit.shape)

apply_feature_optimization(net.decoder, model, latents_to_edit, inds_without)
os.chdir("/home/ayavasileva/sfe-main/launch")
cur_edit_path = f"/home/ayavasileva/data/{feature_generated}_{direction_name}}"

result = subprocess.run(['python', 'fid_calculation.py', f'--orig_path=/home/ayavasileva/data/files_with_{direction_name}3000',
                f'--synt_path={cur_edit_path}',
                            f'--attr_name={celeba_attr}',  '--celeba_attr_table_pth=/home/ayavasileva/3000.txt'],
                            capture_output=True, text=True)

print(result.stdout)
#    print(result.stderr)
fid_score_mid = float(result.stdout.split('\n')[-2].split()[-1])
print(fid_score_mid)
                                                                                     
