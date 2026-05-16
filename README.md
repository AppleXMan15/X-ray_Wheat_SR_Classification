This repository contains the code and experiments for a bachelor's thesis devoted to the study of super-resolution methods for X-ray images of wheat grains and their influence on multi-label defect classification.



\## Project Goal



The main goal of the project is to investigate how image super-resolution affects the quality of automatic classification of internal wheat grain defects from X-ray images.



\## Repository Contents



\- grain segmentation pipeline

\- dataset preparation scripts

\- super-resolution inference scripts

\- classification training and evaluation scripts

\- result visualization

\- experiment summaries



\## Methodology



The project includes the following stages:



1\. Preprocessing of original X-ray images.

2\. Automatic segmentation of individual grains.

3\. Manual annotation of grain defects.

4\. Super-resolution using:

&#x20;  - SRCNN

&#x20;  - EDSR

&#x20;  - ESRGAN

&#x20;  - LDM

&#x20;  - DASR

5\. Training and evaluation of classifiers:

&#x20;  - ResNet

&#x20;  - DenseNet

&#x20;  - EfficientNet

&#x20;  - GoogLeNet

&#x20;  - SqueezeNet

&#x20;  - Vision Transformer

6\. Comparison of results using:

&#x20;  - PSNR

&#x20;  - SSIM

&#x20;  - Precision

&#x20;  - Recall

&#x20;  - F1-score



\## Dataset



The datasets used in this project are stored in cloud storage:



\- \[All Unmarked X-ray images](https://drive.google.com/drive/folders/1lQnqOzhoF\_lGcbVF0\_69k9Y0o50ZEL2Q?usp=drive\_link)

\- \[All Marked Segmented X-ray grains images](https://drive.google.com/drive/folders/1t9sehfDhNuFIIKvL3x2Ch3G0hVpTNmq-?usp=sharing)

\- \[All Pretrained Networks](https://drive.google.com/drive/folders/1mOeCxQ4h-iO5wCl7x93idNXyjZwLV6PZ?usp=sharing)



\## Project Structure



```text

.

├── README.md

├── requirements.txt

├── .gitignore

├── src/

│   ├── segmentation/

│   ├── super\_resolution/

│   └──  classification/

└── figures/

