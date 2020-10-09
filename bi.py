'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob



# Configurations

# GT - Ground-truth;
# Gen: Generated / Restored / Recovered images
p1 = '/media/ml/Data Disk/Kalpesh/NTIRE/valX'
p2 = '/media/ml/Data Disk/Kalpesh/NTIRE/Bic'


img_list = sorted(glob.glob(p1 + '/*'))
#print(img_list)


for i, img_path in enumerate(img_list):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    width = int(img.shape[1] * 4)
    height = int(img.shape[0] * 4)
    dim = (width, height) 
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(p2+base_name+'.png',img_resized)

