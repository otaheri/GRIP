# GRIP: Generating Interaction Poses Using Spatial Cues and Latent Consistency
## Coming Soon - SOS- Still Under Development



[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2308.11617)

![GRAB-Teaser](https://grip.is.tuebingen.mpg.de/media/upload/teaser_final.png)
[[Paper Page](https://grip.is.tue.mpg.de) ] 
[[ArXiv Paper](https://arxiv.org/pdf/2308.11617.pdf) ]


# Video
Check out the YouTube video below for more details.

[![Video](https://github.com/otaheri/GRIP/assets/19238978/a7e20505-7952-4f72-97c5-3d10d4ef633d)
](https://youtu.be/IpIIQrdahYs)


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Getting Started](#getting-started)
  * [Examples](#examples)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)



## Description

This repository Contains:
- Code to preprocess and prepare the GRAB data for the GRIP paper
- Retraining GRIP models, allowing users to change details in the training configuration.
- Code for generating results on the test set split of the data
- Tools to visualize and save generated sequences from GRIP

## Installation

To install the repo please follow the next steps:

- Clone this repository and install the requirements: 
    ```Shell
    git clone https://github.com/otaheri/GRIP
    ```
    ```
    cd GRIP
    pip install -r requirements.txt
    ```

## Getting started
In order to use GRIP please follow the steps below:

- Download the GRAB dataset from [our website](http://grab.is.tue.mpg.de) and make sure to follow the steps there.
- Follow the instructions on the [SMPL-X](https://smpl-x.is.tue.mpg.de) website to download SMPL-X models.
- Check the Examples below to process the required data, ude pretrained GRIP models, and to train GRIP models.


#### data
- Download the GRAB dataset from the [GRAB website](https://grab.is.tue.mpg.de), and follow the instructions there to extract the files.
- Process the data by running the command below.
```commandline
python data/process_data.py --grab-path /path/to/GRAB --smplx-path /path/to/smplx/models/ --out-path /the/output/path
```

#### CNet, RNet, and ANet models
- Please download these models from our website and put them in the folders as below.
```bash
    GRIP
    ├── snapshots
    │   │
    │   ├── anet.pt
    │   ├── cnet.pt
    │   └── rnet.pt
    │   
    │
    .
    .
    .
```

## Examples


- #### Generate hand interaction motion for the test split - CNet and RNet.
    
    ```Shell
    python train/infer_cnet.py --work-dir /path/to/work/dir --grab-path /path/to/GRAB --smplx-path /path/to/models/ --dataset-dir /path/to/processed/data
    ```

- #### Train CNet and RNet with new configurations 
    
    To retrain these models with a new configuration, please use the following code.
    
    ```Shell
    python train/train_cnet.py --work-dir path/to/work/dir --grab-path /path/to/GRAB --smplx-path /path/to/models/ --dataset-dir /path/to/processed/data --expr-id EXPERIMENT_ID

    ```


- #### Generate denoised arm motions and visualize them using ANet on the test split 
    
    ```Shell
    python train/infer_anet.py --work-dir path/to/work/dir --grab-path /path/to/GRAB --smplx-path /path/to/models/ --dataset-dir /path/to/processed/data
    ```


- #### Train ANet for denoising arms with new training configurations 
    
    To retrain ANet with a new configuration, please use the following code.
    
    ```Shell
    python train/train_anet.py --work-dir path/to/work/dir --grab-path /path/to/GRAB --smplx-path /path/to/models/ --dataset-dir /path/to/processed/data --expr-id EXPERIMENT_ID

    ```
    



## Citation

```
@inproceedings{taheri2024grip,
  title  = {{GRIP}: Generating Interaction Poses Using Latent Consistency and Spatial Cues},
  author = {Omid Taheri and Yi Zhou and Dimitrios Tzionas and Yang Zhou and Duygu Ceylan and Soren Pirk and Michael J. Black},
  booktitle = {International Conference on 3D Vision ({3DV})},
  year = {2024},
  url = {https://grip.is.tue.mpg.de}
}
```


## License
Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [LICENSE page](https://grip.is.tue.mpg.de/license.html) for the terms and conditions and any accompanying documentation
before you download and/or use the GRIP data, model and software, (the "Data & Software"),
including 3D meshes (body and objects), images, videos, textures, software, scripts, and animations.
By downloading and/or using the Data & Software (including downloading,
cloning, installing, and any other use of the corresponding github repository),
you acknowledge that you have read and agreed to the LICENSE terms and conditions, understand them,
and agree to be bound by them. If you do not agree with these terms and conditions,
you must not download and/or use the Data & Software. Any infringement of the terms of
this agreement will automatically terminate your rights under this [LICENSE](./LICENSE).


## Acknowledgments
This work was partially supported by Adobe Research (during the first author's internship), the International Max Planck Research School for Intelligent Systems (IMPRS-IS), and the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039B.

We thank:

- Tsvetelina Alexiadis and Alpár Cseke for the Mechanical Turk experiments.
- Benjamin Pellkofer for website design, IT, and web support.


## Contact
The code of this repository was implemented by [Omid Taheri](https://otaheri.github.io/).

For questions, please contact [grip@tue.mpg.de](mailto:grip@tue.mpg.de).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).


