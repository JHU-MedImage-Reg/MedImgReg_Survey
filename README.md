# A survey on deep learning in medical image registration: new technologies, uncertainty, evaluation metrics, and beyond
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2307.15615-b31b1b.svg)](https://arxiv.org/abs/2307.15615)

This official repository contains a comprehensive list of papers on learning-based image registration. Additionally, it includes Python implementations of various image similarity measures, deformation regularization techniques, and evaluation methods for medical image registration.


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
- [Trends in image registration-related research based on PubMed paper counts](#trends-in-image-registration-related-research-based-on-pubmed-paper-counts)

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
* "Unsupervised deformable registration for multi-modal images via disentangled representations", IPMI, 2019 (Qin *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_19)]
* "Adversarial optimization for joint registration and segmentation in prostate CT radiotherapy", MICCAI, 2019 (Elmahdy *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_41)]
* "Adversarial learning for deformable registration of brain MR image using a multi-scale fully convolutional network", BSPC, 2019 (Duan *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1746809419301363)]
* "Adversarial learning for deformable image registration: Application to 3D ultrasound image fusion", SUSI&PIPPI, 2019 (Li and Ogino). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32875-7_7)]
* "Deformable adversarial registration network with multiple loss constraints", CMIG, 2019 (Luo *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S089561112100080X)]
* "Adversarial learning for mono-or multi-modal registration", MedIA, 2019 (Fan *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841519300805)]
* "Adversarial uni-and multi-modal stream networks for multimodal image registration", MICCAI, 2020 (Xu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_22)]
* "Synthesis and inpainting-based MR-CT registration for image-guided thermal ablation of liver tumors", MICCAI, 2019 (Wei *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_57)]
* "SymReg-GAN: symmetric image registration with generative adversarial networks", IEEE TPAMI, 2021 (Zheng *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9440692)]
* "Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance", MedIA, 2022 (Han *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841521003376)]
* "United multi-task learning for abdominal contrast-enhanced CT synthesis through joint deformable registration", CMPB, 2023 (Zhong *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S0169260723000585)]
* "Light-weight deformable registration using adversarial learning with distilling knowledge", IEEE TMI, 2022 (Tran *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9672098)][[GitHub](https://github.com/aioz-ai/LDR_ALDK)]

### Contrastive learning
* "Towards accurate and robust multi-modal medical image registration using contrastive metric learning", IEEE Access, 2019 (Hu *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/8822438)]
* "CoMIR: Contrastive multimodal image representation for registration", NeurIPS, 2020 (Pielawski *et al.*). [[Paper](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html)][[GitHub](https://github.com/MIDA-group/CoMIR)]
* "Can representation learning for multimodal image registration be improved by supervision of intermediate layers?", ICPRIA, 2023 (Wetzer *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-36616-1_21)]
* "Synth-by-reg (sbr): Contrastive learning for synthesis-based registration of paired images", SASHIMI, 2021 (Casamitjana *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87592-3_5)][[GitHub](https://github.com/acasamitjana/SynthByReg)]
* "ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration", MICCAI, 2022 (Dey *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_7)][[GitHub](https://github.com/jmtzt/ContraReg)]
* "Contrastive registration for unsupervised medical image segmentation", IEEE TNNLS, 2023 (Liu *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/10322862)]
* "PC-SwinMorph: Patch representation for unsupervised medical image registration and segmentation", ArXiv, 2022 (Liu *et al.*). [[Paper](https://arxiv.org/abs/2203.05684)]

### Transformers
* "ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration", MIDL, 2021 (Chen *et al.*). [[Paper](https://openreview.net/forum?id=h3HC1EU7AEz)][[GitHub](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)]
* "Transmorph: Transformer for unsupervised medical image registration", MedIA, 2022 (Qin *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002432)][[GitHub](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)]
* "Learning dual transformer network for diffeomorphic registration", MICCAI, 2021 (Zhang *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_13)]
* "Affine Medical Image Registration with Coarse-to-Fine Vision Transformer", CVPR, 2022 (Mok *et al.*). [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Mok_Affine_Medical_Image_Registration_With_Coarse-To-Fine_Vision_Transformer_CVPR_2022_paper.html)][[GitHub](https://github.com/cwmok/C2FViT)]
* "Deformer: Towards displacement field learning for unsupervised medical image registration", MICCAI, 2022 (Chen *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_14)][[GitHub](https://github.com/CJSOrange/DMR-Deformer)]
* "PC-Reg: A pyramidal prediction--correction approach for large deformation image registration", MedIA, 2023 (Yin *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841523002384)]
* "Cross-modal attention for multi-modal image registration", MedIA, 2022 (Song *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002407)][[GitHub](https://github.com/DIAL-RPI/Attention-Reg)]
* "Xmorpher: Full transformer for deformable medical image registration via cross attention", MICCAI, 2022 (Shi *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_21)][[GitHub](https://github.com/Solemoon/XMorpher)]
* "Deformable cross-attention transformer for medical image registration", MLMI, 2023 (Chen *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_12)][[GitHub](https://github.com/junyuchen245/TransMorph_DCA)]
* "TransMatch: a transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration", IEEE TMI, 2023 (Chen *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/10158729)][[GitHub](https://github.com/tzayuan/TransMatch_TMI)]
* "Coordinate translator for learning deformable medical image registration", MMMI, 2022 (Liu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_10)]
* "ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer", MICCAI, 2023 (Wang *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_70)][[GitHub](https://github.com/ZAX130/SmileCode)]
* "Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration", MedIA, 2023 (Khor *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841523000725)]

### Diffusion models
* "DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model", ECCV, 2022 (Kim *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_20)][[GitHub](https://github.com/DiffuseMorph/DiffuseMorph)]
* "FSDiffReg: Feature-Wise and Score-Wise Diffusion-Guided Unsupervised Deformable Image Registration for Cardiac Images", MICCAI, 2023 (Qin and Li). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_62)][[GitHub](https://github.com/xmed-lab/FSDiffReg)]

### Neural ODEs
* "Multi-scale neural ODES for 3D medical image registration", MICCAI, 2021 (Xu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_21)]
* "NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration", CVPR, 2022 (Wu *et al.*). [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_NODEO_A_Neural_Ordinary_Differential_Equation_Based_Optimization_Framework_for_CVPR_2022_paper.pdf)][[GitHub](https://github.com/yifannnwu/NODEO-DIR)]
* "R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks", MedIA, 2023 (Joshi and Hong). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841523001779)][[GitHub](https://github.com/ankitajoshi15/R2Net)]

### Implicit neural representations
* "Diffeomorphic Image Registration With Neural Velocity Field", WACV, 2023 (Han *et al.*). [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Han_Diffeomorphic_Image_Registration_With_Neural_Velocity_Field_WACV_2023_paper.pdf)]
* "Implicit neural representations for deformable image registration", MIDL, 2022 (Wolterink *et al.*). [[Paper](https://openreview.net/forum?id=BP29eKzQBu3)][[GitHub](https://github.com/MIAGroupUT/IDIR)]
* "Implicit neural representations for joint decomposition and registration of gene expression images in the marmoset brain", MICCAI, 2023 (Byra *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_61)][[GitHub](https://github.com/BrainImageAnalysis/ImpRegDec)]
* "Deformable Image Registration with Geometry-informed Implicit Neural Representations", MIDL, 2024 (van Harten *et al.*). [[Paper](https://openreview.net/forum?id=Pj9vtDIzSCE)][[GitHub](https://github.com/Louisvh/tangent_INR_registration_3D)]
* "Topology-preserving shape reconstruction and registration via neural diffeomorphic flow", CVPR, 2022 (Sun *et al.*). [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Topology-Preserving_Shape_Reconstruction_and_Registration_via_Neural_Diffeomorphic_Flow_CVPR_2022_paper.pdf)][[GitHub](https://github.com/Siwensun/Neural_Diffeomorphic_Flow--NDF)]

### Hyperparameter conditioning
* "Learning the Effect of Registration Hyperparameters with HyperMorph", MELBA, 2022 (Hoopes *et al.*). [[Paper](https://arxiv.org/abs/2203.16680)][[GitHub](https://github.com/voxelmorph/voxelmorph/tree/dev)]
* "Conditional deformable image registration with convolutional neural network", MICCAI, 2021 (Mok *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_4)][[GitHub](https://github.com/cwmok/Conditional_LapIRN)]
* "Spatially-varying Regularization with Conditional Transformer for Unsupervised Image Registration", ArXiv, 2023 (Chen *et al.*). [[Paper](https://arxiv.org/abs/2303.06168)][[GitHub](https://github.com/junyuchen245/Spatially_varying_regularization)]

### Anatomy-aware networks
* "Nonuniformly Spaced Control Points Based on Variational Cardiac Image Registration", MICCAI, 2023 (Su *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_60)]
* "A deep discontinuity-preserving image registration network", MICCAI, 2021 (Chen *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_5)][[GitHub](https://github.com/cistib/DDIR)]

### Correlation layer
* "Closing the gap between deep and conventional image registration using probabilistic dense displacement networks", MICCAI, 2019 (Heinrich). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_6)][[GitHub](https://github.com/multimodallearning/pdd_net)]
* "Highly accurate and memory efficient unsupervised learning-based discrete CT registration using 2.5 D displacement search", MICCAI, 2020 (Heinrich and Hansen). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_19)][[GitHub](https://github.com/multimodallearning/pdd2.5/)]
* "Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation", WBIR, 2022 (Heinrich and Hansen). [[Paper](https://openreview.net/pdf?id=SrlgSXA3qAY)][[GitHub](https://github.com/mattiaspaul/VoxelMorphPlusPlus)]
* "Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021", MIDOG, 2021 (Siebert *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-97281-3_25)][[GitHub](https://github.com/multimodallearning/convexAdam)]

### Progressive and pyramid registration
* "Unsupervised 3D end-to-end medical image registration with volume tweening network", IEEE JBHI, 2019 (Zhao *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/8889674/)][[GitHub](https://github.com/microsoft/Recursive-Cascaded-Networks)]
* "Learning a model-driven variational network for deformable image registration", IEEE TMI, 2021 (Jia *et al.*). [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9525092)][[GitHub](https://github.com/xi-jia/Learning-a-Model-Driven-Variational-Network-for-Deformable-Image-Registration)]
* "Unsupervised Learning of Diffeomorphic Image Registration via TransMorph", WBIR, 2022 (Chen *et al.*). [[Paper](https://openreview.net/forum?id=uwIo__2xnTO)][[GitHub](https://github.com/junyuchen245/TransMorph_TVF)]
* "Deep learning-based image registration in dynamic myocardial perfusion CT imaging", IEEE TMI, 2023 (Lara-Hernandez *et al.*). [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9918065)]
* "A multi-scale framework with unsupervised joint training of convolutional neural networks for pulmonary deformable image registration", PMB, 2020 (Jiang *et al.*). [[Paper](https://iopscience.iop.org/article/10.1088/1361-6560/ab5da0/meta)]
* "Dual-stream pyramid registration network", MedIA, 2022 (Kang *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522000317)][[GitHub](https://github.com/kangmiao15/Dual-Stream-PRNet-Plus)]
* "Joint progressive and coarse-to-fine registration of brain MRI via deformation field integration and non-rigid feature fusion", IEEE TMI, 2022 (Lv *et al.*). [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9765391)][[GitHub](https://github.com/JinxLv/Progressvie-and-Coarse-to-fine-Registration-Network)]
* "A deep learning framework for unsupervised affine and deformable image registration", MedIA, 2019 (de Vos *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841518300495)][[GitHub](https://github.com/BDdeVos/TorchIR)]
* "Progressively trained convolutional neural networks for deformable image registration", IEEE TMI, 2019 (Eppenhof *et al.*). [[Paper](https://ieeexplore.ieee.org/document/8902170)]
* "Large deformation diffeomorphic image registration with laplacian pyramid networks", MICCAI, 2020 (Mok *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_21)][[GitHub](https://github.com/cwmok/LapIRN)]
* "Self-recursive contextual network for unsupervised 3D medical image registration", MLMI, 2020 (Hu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59861-7_7)]
* "Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration", IEEE TMI, 2023 (Zhou *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/10042453)][[GitHub](https://github.com/Blcony/SDHNet)]
* "Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration", MICCAI, 2023 (Meng *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_71)][[GitHub](https://github.com/MungoMeng/Registration-NICE-Trans)]
* "PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer", MICCAI, 2023 (Ma *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57)][[GitHub](https://github.com/Torbjorn1997/PIViT)]

## Registration Uncertainty
* "On the applicability of registration uncertainty", MICCAI, 2019 (Luo *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_46)]
* "Double-uncertainty guided spatial and temporal consistency regularization weighting for learning-based abdominal registration", MICCAI, 2022 (Xu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_2)]
* "Transmorph: Transformer for unsupervised medical image registration", MedIA, 2022 (Chen *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002432)][[GitHub](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)]
* "Estimating medical image registration error and confidence: A taxonomy and scoping review", MedIA, 2022 (Bierbrier *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522001785)]
* "From Registration Uncertainty to Segmentation Uncertainty", ISBI, 2024 (Chen *et al.*). [[Paper](https://arxiv.org/abs/2403.05111)][[GitHub](https://github.com/junyuchen245/Registration_Uncertainty)]

## Applications of Image Registration

### Atlas construction
* "Learning conditional deformable templates with convolutional networks", NeurIPS, 2019 (Dalca *et al.*). [[Paper](https://proceedings.neurips.cc/paper/2019/hash/bbcbff5c1f1ded46c25d28119a85c6c2-Abstract.html)][[GitHub](https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/train_cond_template.py)]
* "Generative adversarial registration for improved conditional deformable templates", CVPR, 2021 (Dey *et al.*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Dey_Generative_Adversarial_Registration_for_Improved_Conditional_Deformable_Templates_ICCV_2021_paper.html)][[GitHub](https://github.com/neel-dey/Atlas-GAN)]
* "Unbiased atlas construction for neonatal cortical surfaces via unsupervised learning", ASMUS&PIPPI, 2020 (Cheng *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-60334-2_33)]
* "Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment", CVPR, 2022 (Ding and Niethammer). [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Aladdin_Joint_Atlas_Building_and_Diffeomorphic_Registration_Learning_With_Pairwise_CVPR_2022_paper.pdf)][[GitHub](https://github.com/uncbiag/Aladdin)]
* "Learning 4D infant cortical surface atlas with unsupervised spherical networks", MICCAI, 2021 (Zhao *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_25)]
* "Towards a 4D Spatio-Temporal Atlas of the Embryonic and Fetal Brain Using a Deep Learning Approach for Groupwise Image Registration", WBIR, 2022 (Bastiaansen *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-11203-4_4)]
* "Learning conditional deformable shape templates for brain anatomy", MLMI, 2020 (Yu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59861-7_36)][[GitHub](https://github.com/evanmy/conditional_deformation)]
* "Construction of longitudinally consistent 4D infant cerebellum atlases based on deep learning", MICCAI, 2021 (Chen *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_14)]
* "CAS-Net: conditional atlas generation and brain segmentation for fetal MRI", UNSURE&PIPPI, 2021 (Liu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87735-4_21)]
* "Atlas-ISTN: joint segmentation, registration and atlas construction with image-and-spatial transformer networks", MedIA, 2022 (Sinclair *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522000354)][[GitHub](https://github.com/biomedia-mira/atlas-istn)]
* "Hybrid Atlas Building with Deep Registration Priors", ISBI, 2022 (Wu *et al.*). [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9761670)]
* "Learning-based template synthesis for groupwise image registration", SASHIMI, 2021 (He and Chung). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87592-3_6)]
* "Learning spatiotemporal probabilistic atlas of fetal brains with anatomically constrained registration network", MICCAI, 2021 (Pei *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_23)]
* "ImplicitAtlas: learning deformable shape templates in medical imaging", CVPR, 2022 (Yang *et al.*). [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_ImplicitAtlas_Learning_Deformable_Shape_Templates_in_Medical_Imaging_CVPR_2022_paper.pdf)]
* "GroupRegNet: a groupwise one-shot deep learning-based 4D image registration method", PMB, 2021 (Zhang *et al.*). [[Paper](https://iopscience.iop.org/article/10.1088/1361-6560/abd956)][[GitHub](https://github.com/vincentme/GroupRegNet)]
* "Groupwise Image Registration with Atlas of Multiple Resolutions Refined at Test Phase", MICCAI, 2023 (He *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-47425-5_26)]
* "Learning inverse consistent 3D groupwise registration with deforming autoencoders", SPIE:MI, 2021 (Siebert *et al.*). [[Paper]()][[GitHub]()]

### Multi-atlas segmentation
* "Votenet: A deep learning label fusion method for multi-atlas segmentation", MICCAI, 2019 (Ding *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_23)][[GitHub](https://github.com/uncbiag/VoteNet-Family)]
* "Votenet+: An improved deep learning label fusion method for multi-atlas segmentation", ISBI, 2020 (Ding *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9434031/)][[GitHub](https://github.com/uncbiag/VoteNet-Family)]

## Trends in image registration-related research based on PubMed paper counts
* Learning-based image registration research:
    * Search query: ```("image registration"[Title/Abstract] OR "image alignment"[Title/Abstract]) AND ("Neural Networks"[Title/Abstract] OR "Neural Network"[Title/Abstract] OR "DNN"[Title/Abstract] OR "CNN"[Title/Abstract] OR "ConvNet"[Title/Abstract] OR "Deep Learning"[Title/Abstract] OR "Transformer"[Title/Abstract])```
    * [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/?term=%28%22image+registration%22%5BTitle%2FAbstract%5D+OR+%22image+alignment%22%5BTitle%2FAbstract%5D%29+AND+%28%22Neural+Networks%22%5BTitle%2FAbstract%5D+OR+%22Neural+Network%22%5BTitle%2FAbstract%5D+OR+%22DNN%22%5BTitle%2FAbstract%5D+OR+%22CNN%22%5BTitle%2FAbstract%5D+OR+%22ConvNet%22%5BTitle%2FAbstract%5D+OR+%22Deep+Learning%22%5BTitle%2FAbstract%5D+OR+%22Transformer%22%5BTitle%2FAbstract%5D%29%0D%0A&sort=)
* Unsupervised learning-based image registration research:
    * Search query: ```("Unsupervised"[Title/Abstract] OR "end to end"[Title/Abstract]) AND ("Image Registration"[Title/Abstract] OR "Image Alignment"[Title/Abstract]) AND ("Neural Networks"[Title/Abstract] OR "Neural Network"[Title/Abstract] OR "DNN"[Title/Abstract] OR "CNN"[Title/Abstract] OR "ConvNet"[Title/Abstract] OR "Deep Learning"[Title/Abstract] OR "Transformer"[Title/Abstract])```
    * [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/?term=%28%22Unsupervised%22%5BTitle%2FAbstract%5D+OR+%22end+to+end%22%5BTitle%2FAbstract%5D%29+AND+%28%22Image+Registration%22%5BTitle%2FAbstract%5D+OR+%22Image+Alignment%22%5BTitle%2FAbstract%5D%29+AND+%28%22Neural+Networks%22%5BTitle%2FAbstract%5D+OR+%22Neural+Network%22%5BTitle%2FAbstract%5D+OR+%22DNN%22%5BTitle%2FAbstract%5D+OR+%22CNN%22%5BTitle%2FAbstract%5D+OR+%22ConvNet%22%5BTitle%2FAbstract%5D+OR+%22Deep+Learning%22%5BTitle%2FAbstract%5D+OR+%22Transformer%22%5BTitle%2FAbstract%5D%29%0D%0A&sort=)
