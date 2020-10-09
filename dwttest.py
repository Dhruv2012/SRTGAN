import torch, torchvision
import torchvision.transforms as transforms
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
import glob,os, shutil
import matplotlib.pyplot as plt
from PIL import Image
import pickle
# height=1024, width=768
levels = 2
xfm = DWTForward(J=levels, mode='periodization', wave='bior3.3')
imgs = glob.glob("D:\\upla_sir_stuff\\kadid10k\\images\\???.png")
imgs_n = [i[-7:-4] for i in imgs]
# for i in imgs:
#     os.mkdir("D:\\upla_sir_stuff\\kadid10k\\images\\"+i)
trans = transforms.ToTensor()
imgs = glob.glob("D:\\upla_sir_stuff\\kadid10k\\images\\*\\*.png")
names = [os.path.basename(x) for x in imgs]
print(names)
# print(imgs)
# for i in imgs:
#     img = Image.open(i)
#     img = img.resize((768, 1024),3)
#     img = trans(img).unsqueeze(0)
#     # img = img.permute([0,3,1,2])
#     # print(img.shape)
#     Yl, Yh = xfm(img)
#     print(Yh[0].shape)
#     print(Yh[1].shape)
#     # with open(i[0:-4]+"dwt.pkl", 'wb') as f:
#         # pickle.dump(Yh, f)
#     # print(i[0:-4]+"dwt.pkl")
#     # for j,k in enumerate(Yh):
#     #     dwt = torch.reshape(k, (k.shape[0]*k.shape[1]*k.shape[2], k.shape[3], k.shape[4]))
#     #     torch.save(dwt, i[0:-4]+"dwt"+str(j+1)+".pt")
    

plt.show()
# img = cv2.imread('D:\\upla_sir_stuff\\superres\\original validation\\original validation\\LR\\0801.png')
# img2 = cv2.imread('D:\\upla_sir_stuff\\results\\ICPR\\F1\\801.png')
# img = torch.from_numpy(img)
# img = img/255.0
# img = img[:,:,[2,1,0]]
# img = img.permute(2,0,1)
# img = img.unsqueeze(0)
# img2 = torch.from_numpy(img2)
# img2 = img2/255.0
# img2 = img2[:,:,[2,1,0]]
# img2 = img2.permute(2,0,1)
# img2 = img2.unsqueeze(0)
# Yl, Yh = xfm(img)
# Yl2, Yh2 = xfm(img2)
# print(Yl.shape)
# for i in Yh:
#     i = i.squeeze(0).permute(0,2,3,1)
#     print(i.shape)
#     for j in range(3):
#         plt.imshow(i[j])
#         plt.show()

# qwe = Yh[0][2]
# print(qwe.shape)
# plt.imshow(qwe)
# plt.show()