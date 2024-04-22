# A survey on deep learning in medical image registration: new technologies, uncertainty, evaluation metrics, and beyond
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2307.15615-b31b1b.svg)](https://arxiv.org/abs/2307.15615)

This official repository contains Python implementations of various image similarity measures, deformation regularization techniques, and evaluation methodologies for medical image registration.


## Overview
- [Citation](#citation)
- [Loss functions for image registration](#loss-functions)
    - [Image similarity measures](#image-similarity-measures)
    - [Deformation regularization](#deformation-regularization)
- [Network architectures](#network-architectures)
    - [Conventional ConvNets](#conventional-convnets)
    - [Adversarial learning](#adversarial-learning)
    - [Contrastive learning](#contrastive-learning)
    - [Transformers](#transformers)
    - [Diffusion models](#diffusion-models)
    - [Neural ODEs](#neural-odes)
    - [Implicit neural representations](#implicit-neural-representations)
    - [Hyperparameter conditioning](#hyperparameter-conditioning)
    - [Anatomy-aware networks](#anatomy-aware-networks)
    - [Correlation layer](#correlation-layer)
    - [Progressive and pyramid registration](#progressive-and-pyramid-registration)
- [Registration uncertainty](#registration-uncertainty)
- [Applications of image registration](#applications-of-image-registration)
    - [Atlas construction](#atlas-construction)
    - [Multi-atlas segmentation](#multi-atlas-segmentation)

## Citation
These Python implementations have been prepared for inclusion in the following article:

    @article{chen2023survey,
    title={A survey on deep learning in medical image registration: New technologies, uncertainty, evaluation metrics, and beyond},
    author={Chen, Junyu and Liu, Yihao and Wei, Shuwen and Bian, Zhangxing and Subramanian, Shalini and Carass, Aaron and Prince, Jerry L and Du, Yong},
    journal={arXiv preprint arXiv:2307.15615},
    year={2023}
    }

## Loss Functions

### Image similarity measures
1. Mean Squared Error (MSE)
2. Mean Absolute Error (MAE)
3. Pearson's correlation (PCC) [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L340)
4. Local normalized cross-correlation (LNCC) based on square window [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L22C7-L22C14)
5. Local normalized cross-correlation (LNCC) based on Gaussian window [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L282)
6. Modality independent neighbourhood descriptor (MIND) [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L87)
7. Mutual Information (MI) [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L222)
8. Local mutual information (LMI) [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/0afcd30a7e866aefaf21837130d96a4e17faae91/registration_loss_func/image_sim.py#L282)
9. Correlation Ratio (CR) [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L159)
10. Structural Similarity Index (SSIM) [Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L455)

### Deformation regularization
1. Diffusion regularization
2. Total-variation regularization
3. Bending energy

## Network Architectures

### Conventional ConvNets
* "Nonrigid image registration using multi-scale 3D convolutional neural networks", MICCAI, 2017 (Sokooti *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-66182-7_27)][[GitHub](https://github.com/hsokooti/RegNet/tree/master)]
* "End-to-end unsupervised deformable image registration with a convolutional neural network", DLMIA, 2017 (De Vos *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-67558-9_24)][[GitHub](https://github.com/zhuo-zhi/DIRNet-PyTorch)]
* "Quicksilver: Fast predictive image registration–a deep learning approach", NeuroImage, 2017 (Yang *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1053811917305761)][[GitHub](https://github.com/rkwitt/quicksilver)]
* "Voxelmorph: a learning framework for deformable medical image registration", IEEE TMI, 2017 (Balakrishnan *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/8633930)][[GitHub](https://github.com/voxelmorph/voxelmorph)]
* "Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces", MedIA, 2019 (Dalca *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841519300635)][[GitHub](https://github.com/voxelmorph/voxelmorph)]
* "Deepflash: An efficient network for learning-based medical image registration", CVPR, 2020 (Wang *et al.*). [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_DeepFLASH_An_Efficient_Network_for_Learning-Based_Medical_Image_Registration_CVPR_2020_paper.html)][[GitHub](https://github.com/jw4hv/deepflash)]
* "CycleMorph: cycle consistent unsupervised deformable image registration", MedIA, 2021 (Kim *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841521000827)][[GitHub](https://github.com/boahK/MEDIA_CycleMorph)]
* "U-net vs transformer: Is u-net outdated in medical image registration?", MLMI, 2022 (Jia *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-21014-3_16)][[GitHub](https://github.com/xi-jia/LKU-Net)]

### Adversarial learning
* "Adversarial similarity network for evaluating image alignment in deep learning based registration", MICCAI, 2018 (Fan *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-00928-1_83)]
* "Adversarial image registration with application for MR and TRUS image fusion", MLMI, 2018 (Yan *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-00919-9_23)]
* "Deformable medical image registration using generative adversarial networks", ISBI, 2018 (Mahapatra *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/8363845)]
* "Joint registration and segmentation of xray images using generative adversarial networks", MLMI, 2018 (Mahapatra *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-00919-9_9)]
* "Training data independent image registration using generative adversarial networks and domain adaptation", PR, 2020 (Mahapatra *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320319304108)]
* "Unsupervised deformable registration for multi-modal images via disentangled representations", IPMI, 2019 (Qin *et al.*). [[Paper]()][[GitHub]()]
* "Adversarial optimization for joint registration and segmentation in prostate CT radiotherapy", MICCAI, 2019 (Elmahdy *et al.*). [[Paper]()][[GitHub]()]
* "Adversarial learning for deformable registration of brain MR image using a multi-scale fully convolutional network", BSPC, 2019 (Duan *et al.*). [[Paper]()][[GitHub]()]
* "Adversarial learning for deformable image registration: Application to 3D ultrasound image fusion", SUSI&PIPPI, 2019 (Li and Ogino). [[Paper]()][[GitHub]()]
* "Deformable adversarial registration network with multiple loss constraints", CMIG, 2019 (Luo *et al.*). [[Paper]()][[GitHub]()]
* "Adversarial learning for mono-or multi-modal registration", MedIA, 2019 (Fan *et al.*). [[Paper]()][[GitHub]()]
* "Adversarial uni-and multi-modal stream networks for multimodal image registration", MICCAI, 2020 (Xu *et al.*). [[Paper]()][[GitHub]()]
* "Synthesis and inpainting-based MR-CT registration for image-guided thermal ablation of liver tumors", MICCAI, 2019 (Wei *et al.*). [[Paper]()][[GitHub]()]
* "SymReg-GAN: symmetric image registration with generative adversarial networks", IEEE TPAMI, 2021 (Zheng *et al.*). [[Paper]()][[GitHub]()]
* "Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance", MedIA, 2022 (Han *et al.*). [[Paper]()][[GitHub]()]
* "United multi-task learning for abdominal contrast-enhanced CT synthesis through joint deformable registration", CMPB, 2023 (Zhong *et al.*). [[Paper]()][[GitHub]()]
* "Light-weight deformable registration using adversarial learning with distilling knowledge", IEEE TMI, 2022 (Tran *et al.*). [[Paper]()][[GitHub]()]

### Contrastive learning
* "Towards accurate and robust multi-modal medical image registration using contrastive metric learning", IEEE Access, 2019 (Hu *et al.*). [[Paper]()][[GitHub]()]
* "CoMIR: Contrastive multimodal image representation for registration", NeurIPS, 2020 (Pielawski *et al.*). [[Paper]()][[GitHub]()]
* "Can representation learning for multimodal image registration be improved by supervision of intermediate layers?", ICPRIA, 2023 (Wetzer *et al.*). [[Paper]()][[GitHub]()]
* "Synth-by-reg (sbr): Contrastive learning for synthesis-based registration of paired images", SASHIMI, 2021 (Casamitjana *et al.*). [[Paper]()][[GitHub]()]
* "ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration", MICCAI, 2022 (Dey *et al.*). [[Paper]()][[GitHub]()]
* "Contrastive registration for unsupervised medical image segmentation", IEEE TNNLS, 2023 (Liu *et al.*). [[Paper]()][[GitHub]()]
* "PC-SwinMorph: Patch representation for unsupervised medical image registration and segmentation", ArXiv, 2022 (Liu *et al.*). [[Paper]()][[GitHub]()]

### Transformers
* "ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration", MIDL, 2021 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "Transmorph: Transformer for unsupervised medical image registration", MedIA, 2022 (Qin *et al.*). [[Paper]()][[GitHub]()]
* "Learning dual transformer network for diffeomorphic registration", MICCAI, 2021 (Zhang *et al.*). [[Paper]()][[GitHub]()]
* "Affine Medical Image Registration with Coarse-to-Fine Vision Transformer", CVPR, 2022 (Mok *et al.*). [[Paper]()][[GitHub]()]
* "Deformer: Towards displacement field learning for unsupervised medical image registration", MICCAI, 2022 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "PC-Reg: A pyramidal prediction--correction approach for large deformation image registration", MedIA, 2023 (Yin *et al.*). [[Paper]()][[GitHub]()]
* "Cross-modal attention for multi-modal image registration", MedIA, 2022 (Song *et al.*). [[Paper]()][[GitHub]()]
* "Xmorpher: Full transformer for deformable medical image registration via cross attention", MICCAI, 2022 (Shi *et al.*). [[Paper]()][[GitHub]()]
* "Deformable cross-attention transformer for medical image registration", MLMI, 2023 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "TransMatch: a transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration", IEEE TMI, 2023 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "Coordinate translator for learning deformable medical image registration", MMMI, 2022 (Liu *et al.*). [[Paper]()][[GitHub]()]
* "ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer", MICCAI, 2023 (Wang *et al.*). [[Paper]()][[GitHub]()]
* "Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration", MedIA, 2023 (Khor *et al.*). [[Paper]()][[GitHub]()]

### Diffusion models
* "DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model", ECCV, 2022 (Kim *et al.*). [[Paper]()][[GitHub]()]
* "FSDiffReg: Feature-Wise and Score-Wise Diffusion-Guided Unsupervised Deformable Image Registration for Cardiac Images", MICCAI, 2023 (Qin and Li). [[Paper]()][[GitHub]()]

### Neural ODEs
* "Multi-scale neural ODES for 3D medical image registration", MICCAI, 2021 (Xu *et al.*). [[Paper]()][[GitHub]()]
* "NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration", CVPR, 2022 (Wu *et al.*). [[Paper]()][[GitHub]()]
* "R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks", MedIA, 2023 (Joshi and Hong). [[Paper]()][[GitHub]()]

### Implicit neural representations
* "Diffeomorphic Image Registration With Neural Velocity Field", WACV, 2023 (Han *et al.*). [[Paper]()][[GitHub]()]
* "Implicit neural representations for deformable image registration", MIDL, 2022 (Wolterink *et al.*). [[Paper]()][[GitHub]()]
* "Implicit neural representations for joint decomposition and registration of gene expression images in the marmoset brain", MICCAI, 2023 (Byra *et al.*). [[Paper]()][[GitHub]()]
* "Deformable Image Registration with Geometry-informed Implicit Neural Representations", MIDL, 2024 (van Harten *et al.*). [[Paper]()][[GitHub]()]
* "Topology-preserving shape reconstruction and registration via neural diffeomorphic flow", CVPR, 2022 (Sun *et al.*). [[Paper]()][[GitHub]()]

### Hyperparameter conditioning
* "Learning the Effect of Registration Hyperparameters with HyperMorph", MELBA, 2022 (Hoopes *et al.*). [[Paper]()][[GitHub]()]
* "Conditional deformable image registration with convolutional neural network", MICCAI, 2021 (Mok *et al.*). [[Paper]()][[GitHub]()]
* "Spatially-varying Regularization with Conditional Transformer for Unsupervised Image Registration", ArXiv, 2023 (Chen *et al.*). [[Paper]()][[GitHub]()]

### Anatomy-aware networks
* "Nonuniformly Spaced Control Points Based on Variational Cardiac Image Registration", MICCAI, 2023 (Su *et al.*). [[Paper]()][[GitHub]()]
* "A deep discontinuity-preserving image registration network", MICCAI, 2021 (Chen *et al.*). [[Paper]()][[GitHub]()]

### Correlation layer
* "Closing the gap between deep and conventional image registration using probabilistic dense displacement networks", MICCAI, 2019 (Heinrich). [[Paper]()][[GitHub]()]
* "Highly accurate and memory efficient unsupervised learning-based discrete CT registration using 2.5 D displacement search", MICCAI, 2020 (Heinrich and Hansen). [[Paper]()][[GitHub]()]
* "Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation", WBIR, 2022 (Heinrich and Hansen). [[Paper]()][[GitHub]()]
* "Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021", MIDOG, 2021 (Siebert *et al.*). [[Paper]()][[GitHub]()]

### Progressive and pyramid registration
* "Unsupervised 3D end-to-end medical image registration with volume tweening network", IEEE JBHI, 2019 (Zhao *et al.*). [[Paper]()][[GitHub]()]
* "Learning a model-driven variational network for deformable image registration", IEEE TMI, 2021 (Jia *et al.*). [[Paper]()][[GitHub]()]
* "Unsupervised Learning of Diffeomorphic Image Registration via TransMorph", WBIR, 2022 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "Deep learning-based image registration in dynamic myocardial perfusion CT imaging", IEEE TMI, 2023 (Lara-Hernandez *et al.*). [[Paper]()][[GitHub]()]
* "A multi-scale framework with unsupervised joint training of convolutional neural networks for pulmonary deformable image registration", PMB, 2020 (Jiang *et al.*). [[Paper]()][[GitHub]()]
* "Dual-stream pyramid registration network", MedIA, 2022 (Kang *et al.*). [[Paper]()][[GitHub]()]
* "Joint progressive and coarse-to-fine registration of brain MRI via deformation field integration and non-rigid feature fusion", IEEE TMI, 2022 (Lv *et al.*). [[Paper]()][[GitHub]()]
* "A deep learning framework for unsupervised affine and deformable image registration", MedIA, 2019 (de Vos *et al.*). [[Paper]()][[GitHub]()]
* "Progressively trained convolutional neural networks for deformable image registration", IEEE TMI, 2019 (Eppenhof *et al.*). [[Paper]()][[GitHub]()]
* "Large deformation diffeomorphic image registration with laplacian pyramid networks", MICCAI, 2020 (Mok *et al.*). [[Paper]()][[GitHub]()]
* "Self-recursive contextual network for unsupervised 3D medical image registration", MLMI, 2020 (Hu *et al.*). [[Paper]()][[GitHub]()]
* "Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration", IEEE TMI, 2023 (Zhou *et al.*). [[Paper]()][[GitHub]()]
* "Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration", MICCAI, 2023 (Meng *et al.*). [[Paper]()][[GitHub]()]
* "PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer", MICCAI, 2023 (Ma *et al.*). [[Paper]()][[GitHub]()]

## Registration Uncertainty
* "On the applicability of registration uncertainty", MICCAI, 2019 (Luo *et al.*). [[Paper]()][[GitHub]()]
* "Double-uncertainty guided spatial and temporal consistency regularization weighting for learning-based abdominal registration", MICCAI, 2022 (Xu *et al.*). [[Paper]()][[GitHub]()]
* "Transmorph: Transformer for unsupervised medical image registration", MedIA, 2022 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "Estimating medical image registration error and confidence: A taxonomy and scoping review", MedIA, 2022 (Bierbrier *et al.*). [[Paper]()][[GitHub]()]
* "From Registration Uncertainty to Segmentation Uncertainty", ISBI, 2024 (Chen *et al.*). [[Paper]()][[GitHub]()]

## Applications of Image Registration

### Atlas construction
* "Learning conditional deformable templates with convolutional networks", NeurIPS, 2019 (Dalca *et al.*). [[Paper]()][[GitHub]()]
* "Generative adversarial registration for improved conditional deformable templates", CVPR, 2021 (Dey *et al.*). [[Paper]()][[GitHub]()]
* "Unbiased atlas construction for neonatal cortical surfaces via unsupervised learning", ASMUS&PIPPI, 2020 (Cheng *et al.*). [[Paper]()][[GitHub]()]
* "Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment", CVPR, 2022 (Ding and Niethammer). [[Paper]()][[GitHub]()]
* "Learning 4D infant cortical surface atlas with unsupervised spherical networks", MICCAI, 2021 (Zhao *et al.*). [[Paper]()][[GitHub]()]
* "Towards a 4D Spatio-Temporal Atlas of the Embryonic and Fetal Brain Using a Deep Learning Approach for Groupwise Image Registration", WBIR, 2022 (Bastiaansen *et al.*). [[Paper]()][[GitHub]()]
* "Learning conditional deformable shape templates for brain anatomy", MLMI, 2020 (Yu *et al.*). [[Paper]()][[GitHub]()]
* "Construction of longitudinally consistent 4D infant cerebellum atlases based on deep learning", MICCAI, 2021 (Chen *et al.*). [[Paper]()][[GitHub]()]
* "CAS-Net: conditional atlas generation and brain segmentation for fetal MRI", UNSURE&PIPPI, 2021 (Liu *et al.*). [[Paper]()][[GitHub]()]
* "Atlas-ISTN: joint segmentation, registration and atlas construction with image-and-spatial transformer networks", MedIA, 2022 (Sinclair *et al.*). [[Paper]()][[GitHub]()]
* "Hybrid Atlas Building with Deep Registration Priors", ISBI, 2022 (Wu *et al.*). [[Paper]()][[GitHub]()]
* "Learning-based template synthesis for groupwise image registration", SASHIMI, 2021 (He and Chung). [[Paper]()][[GitHub]()]
* "Learning spatiotemporal probabilistic atlas of fetal brains with anatomically constrained registration network", MICCAI, 2021 (Pei *et al.*). [[Paper]()][[GitHub]()]
* "ImplicitAtlas: learning deformable shape templates in medical imaging", CVPR, 2022 (Yang *et al.*). [[Paper]()][[GitHub]()]
* "GroupRegNet: a groupwise one-shot deep learning-based 4D image registration method", PMB, 2021 (Zhang *et al.*). [[Paper]()][[GitHub]()]
* "Groupwise Image Registration with Atlas of Multiple Resolutions Refined at Test Phase", MICCAI, 2023 (He *et al.*). [[Paper]()][[GitHub]()]
* "Learning inverse consistent 3D groupwise registration with deforming autoencoders", SPIE:MI, 2021 (Siebert *et al.*). [[Paper]()][[GitHub]()]

### Multi-atlas segmentation
* "Votenet: A deep learning label fusion method for multi-atlas segmentation", MICCAI, 2019 (Ding *et al.*). [[Paper]()][[GitHub]()]
* "Votenet+: An improved deep learning label fusion method for multi-atlas segmentation", ISBI, 2020 (Ding *et al.*). [[Paper]()][[GitHub]()]
