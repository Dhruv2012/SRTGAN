# **SRTGAN: Triplet Loss based Generative Adversarial Network for Real-World Super-Resolutions**  [[Website](https://srtgan.github.io/)][[Paper](https://arxiv.org/abs/2211.12180.pdf)]
![SRTGAN](readme_images/ProposedArchitecture.png)

A high level overview along with architecture and description of different scripts can be found here: [Description](Description.md)

## Abstract
Many applications such as forensics, surveillance, satellite imaging, medical imaging, etc., demand High-Resolution (HR) images. However, obtaining an HR image is not always possible due to the limitations of optical sensors and their costs. An alternative solution called Single Image Super-Resolution (SISR) is a software-driven approach that aims to take a Low-Resolution (LR) image and obtain the HR image. Most supervised SISR solutions use ground truth HR image as a target and do not include the information provided in the LR image, which could be valuable. In this work, we introduce Triplet Loss-based Generative Adversarial Network hereafter referred as SRTGAN for Image Super-Resolution problem on real-world degradation. We introduce a new triplet-based adversarial loss function that exploits the information provided in the LR image by using it as a negative sample. Allowing the patch-based discriminator with access to both HR and LR images optimizes to better differentiate between HR and LR images; hence, improving the adversary. Further, we propose to fuse the adversarial loss, content loss, perceptual loss, and quality loss to obtain Super-Resolution (SR) image with high perceptual fidelity. We validate the superior performance of the proposed method over the other existing methods on the RealSR dataset in terms of quantitative and qualitative metrics.

## Publications
Presented at the 7th International Conference on Computer Vision and Image Processing

## Citing us
```
@misc{https://doi.org/10.48550/arxiv.2211.12180,
  doi = {10.48550/ARXIV.2211.12180},
  
  url = {https://arxiv.org/abs/2211.12180},
  
  author = {Patel, Dhruv and Jain, Abhinav and Bawkar, Simran and Khorasiya, Manav and Prajapati, Kalpesh and Upla, Kishor and Raja, Kiran and Ramachandra, Raghavendra and Busch, Christoph},
  
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {SRTGAN: Triplet Loss based Generative Adversarial Network for Real-World Super-Resolution},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Setup

1. Open any terminal and install all required dependencies:

   ```
   pip install -r requirements.txt
   ```
2. Edit the Dataset paths in `options/train/train_srtgan.json` and `options/test/test_srtgan.json`.

## Training
To train our model, run the following command

```
python train.py -opt options/train/train_srtgan.json
```

## Testing
For testing the trained model, run the following command

```
python test.py -opt options/test/test_srtgan.json
```

## Model Weights
Following are the drive links for model weight files:
 * [Generator](https://drive.google.com/file/d/1GfXis8UK3oLmVyhHtr2ZyFHdSCtQQGVz/view?usp=share_link)
 * [Discriminator](https://drive.google.com/file/d/11Yq9deEx6RgnZ1JrsPmRrFvot4IsG8AW/view?usp=share_link)
 * [State-file](https://drive.google.com/file/d/1x3T_lz7j_o_VLGYG4eOXya6q1xSKSj59/view?usp=sharing)

# Table of Results
The quantitative assessment of the proposed method without QA and without discriminator networks carried out on RealSR validation dataset.
<p align="center">
   <img src="./readme_images/result_table1.png" height="75">
</p>

The quantitative comparison of the proposed and other existing SR methods on Real SR validation and DIV2KRK datasets.
<p align="center">
   <img src="./readme_images/result_table2.png" height="200">
</p>

# Output 
The comparison of the Super-Resolution results obtained using the Proposed Architecture and Without Quality Loss method and Without Triplet Loss (Vanilla GAN Loss) method on NTIRE-2020 Real world SR  challenge testing dataset.
<p align="center">
   <img src="./readme_images/results.png" height="300">
</p>

## References
For designing our Code Framework, we have taken reference of the following repository - https://github.com/xinntao/BasicSR

