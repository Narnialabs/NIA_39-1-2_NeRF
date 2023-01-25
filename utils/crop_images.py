import os, glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import argparse

def crop_im(load_path,img_res = 800):
    if 'Mask' in load_path: 
        img_load  = load_path + 'mask_view{}.png'.format(str(i).zfill(2))
    
    else:
        img_load  = load_path + 'image_view{}.png'.format(str(i).zfill(2))
    
    img = Image.open(img_load)
    img = np.asarray(img) 
        
    H,W =img.shape[:2]
    pad_H,pad_W = (H-img_res)//2, (W-img_res)//2
    img = img[pad_H:H-pad_H,pad_W:W-pad_W,:]
    
    W, H = 800,800
    imgs_reszed_res = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    
    return imgs_reszed_res


def black_bgr():
    shape = mask_img.shape[:2]
    arr = np.zeros([shape[0],shape[1],3])
    x,y = np.where(mask_img[:,:,0]!=0)
    arr[x,y,0] = rgb_img[x,y,0]
    x,y = np.where(mask_img[:,:,1]!=0)
    arr[x,y,1] = rgb_img[x,y,1]
    x,y = np.where(mask_img[:,:,2]!=0)
    arr[x,y,2] = rgb_img[x,y,2]
    arr = arr*255.
    return arr2Img(arr)

arr2Img = lambda img : Image.fromarray(img.astype('uint8'))


def get_parser():
    #configs:
    parser = argparse.ArgumentParser(description='For Nia-39-1 NeRF image processing')
    parser.add_argument('--src_path',type=str)
    parser.add_argument('--dst_path',type=str,default='./dataset/data_square/')
    parser.add_argument('--tgt_asset',type=str)
    parser.add_argument('--config')
    return parser

if __name__ == '__main__':
    
    
    parser = get_parser()
    args = parser.parse_args()
    
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        params = config['parameters']

        for key,value in params.items():
            globals()[args.key] = value
            
            
    load_path_rgb = args.src_path + args.tgt_asset + '/RGB img/'
    load_path_mask = args.src_path + args.tgt_asset + '/Mask/'

    save_path = args.dst_path + args.tgt_asset + '/'
    os.makedirs(save_path,exist_ok=True)

    print('...Start cropping img : {}'.format(args.tgt_asset))

    for i in range(24):
        mask_img = crop_im(load_path_mask,img_res=800)/255.
        rgb_img = crop_im(load_path_rgb,img_res=800)/255.
        print(i,rgb_img.shape)
        new_img = black_bgr()        
        img_save = save_path + 'image_view{}.png'.format(str(i).zfill(2))
        new_img.save(img_save)
            
            