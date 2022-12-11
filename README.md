# Project Description

A high level overview along with architecture and description of different scripts can be found here: [Description](Description.md)

# Setup

1. Open any terminal and install all required dependencies:

   ```
   pip install -r requirements.txt
   ```
2. Edit the Dataset paths in `options/train/train_srtgan.json` and `options/test/test_srtgan.json`.

# Training
To train our model, run the following command

```
python train.py -opt options/train/train_srtgan.json
```

# Testing
For testing the trained model, run the following command

```
python test.py -opt options/test/test_srtgan.json
```

# References
For designing our Code Framework, we have taken reference of the following repository - https://github.com/xinntao/BasicSR

# Publications
[Arxiv Link](https://arxiv.org/abs/2211.12180)

# Website
A simple description of our project: [SRTGAN](https://srtgan.github.io/srtgan/)

