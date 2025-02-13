# A survey on deep learning in medical image registration: new technologies, uncertainty, evaluation metrics, and beyond
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2307.15615-b31b1b.svg)](https://arxiv.org/abs/2307.15615)

This official repository contains a comprehensive list of papers on learning-based image registration. Additionally, it includes Python implementations of various image similarity measures, deformation regularization techniques, and evaluation methods for medical image registration.

$${\color{red}New!}$$ 10/04/2024 - Our paper has been accepted by ***Medical Image Analysis*** for publication! [[Link](https://www.sciencedirect.com/science/article/pii/S1361841524003104)]

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
- [Benchmark dataset for medical image registration](#Benchmark-dataset-for-medical-image-registration)
- [Applications of image registration](#applications-of-image-registration)
    - [Atlas construction](#atlas-construction)
    - [Multi-atlas segmentation](#multi-atlas-segmentation)
    - [Uncertainty](#uncertainty)
    - [Motion estimation](#motion-estimation)
    - [2D-3D registration](#2D-3D-registration)
- [Towards Zero-shot Registration (Fundation Models)](#towards-zero-shot-registration-or-fundation-models)
- [Trends in image registration-related research based on PubMed paper counts](#trends-in-image-registration-related-research-based-on-pubmed-paper-counts)

## Citation
These Python implementations and the list of papers have been prepared for inclusion in the following article:

    @article{chen2023survey,
    title = {A survey on deep learning in medical image registration: New technologies, uncertainty, evaluation metrics, and beyond},
    author = {Junyu Chen and Yihao Liu and Shuwen Wei and Zhangxing Bian and Shalini Subramanian and Aaron Carass and Jerry L. Prince and Yong Du},
    journal = {Medical Image Analysis},
    pages = {103385},
    year = {2024},
    issn = {1361-8415},
    doi = {https://doi.org/10.1016/j.media.2024.103385}
    }

## Loss Functions

### Image similarity measures
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Pearson's correlation (PCC) [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L340)]
* Local normalized cross-correlation (LNCC) based on square window [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L22C7-L22C14)]
* Local normalized cross-correlation (LNCC) based on Gaussian window [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L282)]
* Modality independent neighbourhood descriptor (MIND) [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L87)]
* Mutual Information (MI) [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L222)]
* Local mutual information (LMI) [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/0afcd30a7e866aefaf21837130d96a4e17faae91/registration_loss_func/image_sim.py#L282)]
* Structural Similarity Index (SSIM) [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/567da0b653e2be3ebe8909dd978ef83c247c16f7/registration_loss_func/image_sim.py#L455)]
* Normalized Gradient Fields (NGF) [[GitHub](https://github.com/BailiangJ/learn2reg2021_task3/blob/master/normalized_gradient_field.py)]
* $${\color{red}New!}$$ Correlation Ratio (CR) [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/766deb50b1c3b99d7fceca05cb3508359d59a7f9/registration_loss_func/image_sim.py#L543)][[Paper](https://arxiv.org/abs/2409.13863)]
* $${\color{red}New!}$$ Local correlation ratio [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/766deb50b1c3b99d7fceca05cb3508359d59a7f9/registration_loss_func/image_sim.py#L612)][[Paper](https://arxiv.org/abs/2409.13863)]

### Deformation regularization
* Diffusion regularization [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/510d47ba692daa032cda9b60c4d704ab5a398ae7/registration_loss_func/deformation_regularizer.py#L22), use `penalty='l2'`]
* Total-Variation regularization [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/510d47ba692daa032cda9b60c4d704ab5a398ae7/registration_loss_func/deformation_regularizer.py#L22), use `penalty='l1'`]
* Isotropic Total-Variation regularization [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/510d47ba692daa032cda9b60c4d704ab5a398ae7/registration_loss_func/deformation_regularizer.py#L49)]
* Bending energy [[Code](https://github.com/JHU-MedImage-Reg/MedImgReg_Survey/blob/510d47ba692daa032cda9b60c4d704ab5a398ae7/registration_loss_func/deformation_regularizer.py#L71)]
* ICON
    * "ICON: Learning Regular Maps Through Inverse Consistency", ICCV, 2021 (Greer *et al.*). [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Greer_ICON_Learning_Regular_Maps_Through_Inverse_Consistency_ICCV_2021_paper.pdf)][[GitHub](https://github.com/uncbiag/ICON)]
* GradICON
    * "GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency", CVPR, 2022 (Tian *et al.*). [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tian_GradICON_Approximate_Diffeomorphisms_via_Gradient_Inverse_Consistency_CVPR_2023_paper.pdf)][[GitHub](https://github.com/uncbiag/ICON)]
* Inverse consistency by construction
    * "Inverse consistency by construction for multistep deep registration", MICCAI, 2023 (Greer *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_65)][[GitHub](https://github.com/uncbiag/ByConstructionICON)]
* $${\color{red}New!}$$ Spatially varying regularization
    * "Unsupervised learning of spatially varying regularization for diffeomorphic image registration", arXiv, 2024 (Chen *et al.*). [[Paper](https://arxiv.org/abs/2412.17982)][[GitHub](https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg)]

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
* $${\color{red}New!}$$ "ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration", IEEE TMI, 2024 (Siebert *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/10681158)][[GitHub](https://github.com/multimodallearning/convexAdam)]

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

## Benchmark dataset for medical image registration
$${\color{red}New!}$$
| Dataset | Anatomy | Cohort Type | Modality | Source |
|---|---|---|---|---|
| IXI | Brain | Healthy Controls | T1w, T2w, PDw MRI | [Official Website](https://brain-development.org/ixi-dataset/) |
| LUMIR | Brain | Healthy Controls | T1w MRI | [Learn2Reg 2024](https://learn2reg.grand-challenge.org/) |
| LPBA40 | Brain | Healthy Controls | T1w MRI | [Official Website](https://www.loni.usc.edu/research/atlases) |
| Mindboggle | Brain | Healthy Controls | T1w MRI | [Official Website](https://mindboggle.info/) |
| OASIS | Brain | Alzheimer’s disease | T1w MRI | [Official Website](https://sites.wustl.edu/oasisbrains/) |
| BraTS-Reg | Brain | Glioma | T1w, T1ce, T2w, FLAIR MRI | [Official Website](https://www.med.upenn.edu/cbica/brats-reg-challenge/) |
| CuRIOUS | Brain | Glioma | T1w, T2-FLAIR MRI, 3D US | [Learn2Reg 2020](https://learn2reg.grand-challenge.org/) |
| ReMIND2Reg | Brain | Tumor resection | T1w, T2w MRI, 3D US | [Learn2Reg 2024](https://learn2reg.grand-challenge.org/) |
| Hippocampus-MR | Brain | Non-affective psychosis | T1w MRI | [Learn2Reg 2020](https://learn2reg.grand-challenge.org/) |
| DIR-Lab | Lung | COPD, cancer | Breath-hold and 4DCT | [Official Website](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/index.html) |
| NLST | Lung | Smokers | Spiral CT | [Official Website](view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.cancerimagingarchive.net%2Fwp-content%2Fuploads%2FINFOclinical_HN_Version2_30may2018.xlsx&wdOrigin=BROWSELINK) |
| Lung-CT | Lung | Healthy Controls | Inspiratory, expiratory CT | [Learn2Reg 2021](https://learn2reg.grand-challenge.org/) |
| EMPIRE10 | Lung | Healthy Controls | Inspiratory, expiratory CT | [Official Website](https://empire10.grand-challenge.org/) |
| Thorax-CBCT | Lung | Cancer Patients | CT, CBCT | [Learn2Reg 2023](https://learn2reg.grand-challenge.org/) |
| Lung250M-4B | Lung | Mixed | CT | [Official Website](https://github.com/multimodallearning/Lung250M-4B) |
| ACDC | Heart | Cardiac diseases | 4D cine-MRI | [Official Website](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) |
| M&Ms | Heart | Cardiac diseases | 4D cine-MRI | [Official Website](https://www.ub.edu/mnms/) |
| MM-WHS | Heart | Cardiac diseases | CT, MRI | [Official Website](https://zmiclab.github.io/zxh/0/mmwhs/) |
| Abdomen-CT-CT | Abdomen | Cancer Patients | CT | [Learn2Reg 2020](https://learn2reg.grand-challenge.org/) |
| Abdomen-MR-CT | Abdomen | Cancer Patients | CT, MR | [Learn2Reg 2021](https://learn2reg.grand-challenge.org/) |
| ACROBAT | Breast | Breast Cancer | Pathological images | [Official Website](https://acrobat.grand-challenge.org/) |
| ANHIR | Body-wide | Cancer tissue samples | Pathological images | [Official Website](https://anhir.grand-challenge.org/) |
| COMULISglobe SHG-BF | Breast / Pancreas | Cancer tissue samples | Pathological images | [Learn2Reg 2024](https://learn2reg.grand-challenge.org/) |
| COMULISglobe 3D-CLEM | Cell | Mitochondria, nuclei | Microscopy | [Learn2Reg 2024](https://learn2reg.grand-challenge.org/) |

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
* "Learning inverse consistent 3D groupwise registration with deforming autoencoders", SPIE:MI, 2021 (Siebert *et al.*). [[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11596/115960F/Learning-inverse-consistent-3D-groupwise-registration-with-deforming-autoencoders/10.1117/12.2581948.short#_=_)]
* "Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers", NeurIPS, 2022 (Wang and Zhang). [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b328c5bd9ff8e3a5e1be74baf4a7a456-Paper-Conference.pdf)][[GitHub](https://github.com/jw4hv/Geo-SIC)]

### Multi-atlas segmentation
* "Votenet: A deep learning label fusion method for multi-atlas segmentation", MICCAI, 2019 (Ding *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_23)][[GitHub](https://github.com/uncbiag/VoteNet-Family)]
* "Votenet+: An improved deep learning label fusion method for multi-atlas segmentation", ISBI, 2020 (Ding *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9434031/)][[GitHub](https://github.com/uncbiag/VoteNet-Family)]
* "Cross-Modality Multi-Atlas Segmentation via Deep Registration and Label Fusion", IEEE JBHI, 2022 (Ding *et al.*). [[Paper]()][[GitHub]()]
* "Multi-atlas segmentation and spatial alignment of the human embryo in first trimester 3D ultrasound", MELBA, 2022 (Bastiaansen *et al.*). [[Paper]()][[GitHub]()]
* "Atlas-ISTN: joint segmentation, registration and atlas construction with image-and-spatial transformer networks", MedIA, 2022 (Sinclair *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522000354)][[GitHub](https://github.com/biomedia-mira/atlas-istn)]
* "Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration", MedIA, 2023 (Khor *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841523000725)]
* "DeepAtlas: Joint semi-supervised learning of image registration and segmentation", MICCAI, 2019 (Xu and Niethammer). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_47)][[GitHub](https://github.com/uncbiag/DeepAtlas)]

### Uncertainty
* "Deformable image registration uncertainty for inter-fractional dose accumulation of lung cancer proton therapy", RO, 2020 (Nenoff *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S0167814020302279)]

### Motion estimation
* "Joint learning of motion estimation and segmentation for cardiac MR image sequences", MICCAI, 2018 (Qin *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_53)][[GitHub](https://github.com/cq615/Joint-Motion-Estimation-and-Segmentation)]
* "Implementation and validation of a three-dimensional cardiac motion estimation network", Radiology:AI, 2019 (Morales *et al.*). [[Paper](https://pubs.rsna.org/doi/full/10.1148/ryai.2019180080)]
* "MulViMotion: Shape-aware 3D Myocardial Motion Tracking from Multi-View Cardiac MRI", IEEE TMI, 2022 (Meng *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9721301/)][[GitHub](https://github.com/ImperialCollegeLondon/Multiview-Motion-Estimation-for-3D-cardiac-motion-tracking)]
* "FOAL: Fast online adaptive learning for cardiac motion estimation", CVPR, 2020 (Yu *et al.*). [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_FOAL_Fast_Online_Adaptive_Learning_for_Cardiac_Motion_Estimation_CVPR_2020_paper.html)]
* "Generative myocardial motion tracking via latent space exploration with biomechanics-informed prior", MedIA, 2023 (Qin *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522003103)][[GitHub](https://github.com/cq615/BIGM-motion-tracking)]
* "WarpPINN: Cine-MR image registration with physics-informed neural networks", MedIA, 2022 (L{\'o}pez *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841523001858?casa_token=z2ON3FdOg9MAAAAA:P5tHxdo_A6BL9_JQTiAspydEztlBLpFbdPYt-8HRPPe0cMeJ4nsfLZk9ztM_7sQtBGH0Phg)][[GitHub](https://github.com/fsahli/WarpPINN)]
* "DeepTag: An unsupervised deep learning method for motion tracking on cardiac tagging magnetic resonance images", CVPR, 2021 (Ye *et al.*). [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Ye_DeepTag_An_Unsupervised_Deep_Learning_Method_for_Motion_Tracking_on_CVPR_2021_paper.html)][[GitHub](https://github.com/DeepTag/cardiac_tagging_motion_estimation)]
* "DRIMET: Deep registration-based 3D incompressible motion estimation in Tagged-MRI with application to the tongue", MIDL, 2024 (Bian *et al.*). [[Paper](https://proceedings.mlr.press/v227/bian24a)][[GitHub](https://github.com/jasonbian97/DRIMET-tagged-MRI)]
* "Momentamorph: Unsupervised spatial-temporal registration with momenta, shooting, and correction", MICCAI, 2023 (Bian *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-47425-5_3)]
* "A semi-supervised joint network for simultaneous left ventricular motion tracking and segmentation in 4D echocardiography", MICCAI, 2020 (Ta *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_45)]
* "Unsupervised motion tracking of left ventricle in echocardiography", SPIE:MI, 2020 (Ahn *et al.*). [[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11319/113190Z/Unsupervised-motion-tracking-of-left-ventricle-in-echocardiography/10.1117/12.2549572.short#_=_)]
* "LungRegNet: an unsupervised deformable image registration method for 4D-CT lung", Med. Phys., 2020 (Fu *et al.*). [[Paper](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.14065?casa_token=va2zLS3Z1bAAAAAA%3AuleEwqwSWXJ-a3xzLTLtivIjPWacKZ_Ic_1LYlQI0ROUF57SW641ZBJSJduKOQmpRU-sI_bFDOwk)]
* "An unsupervised image registration method employing chest computed tomography images and deep neural networks", CBM, 2023 (Ho *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S001048252300077X?casa_token=N0YXVGEhAGIAAAAA:mJ8RAejLvf3rlJpYop3wKRxb2_aLgddPhVJIZQXPi9uqqIuj9jn7KXvUqn3k9mWB88Y4vvo)]
* "One-shot learning for deformable medical image registration and periodic motion tracking", IEEE TMI, 2020 (Fechter and Baltas). [[Paper](https://ieeexplore.ieee.org/abstract/document/8989991)][[GitHub](https://github.com/ToFec/OneShotImageRegistration)]
* "CNN-based lung CT registration with multiple anatomical constraints", MedIA, 2021 (Hering *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841521001857)][[Code](https://grand-challenge.org/algorithms/deep-learning-based-ct-lung-registration/)]
* "A One-shot Lung 4D-CT Image Registration Method with Temporal-spatial Features", BioCAS, 2022 (Ji *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9948656)]
* "ORRN: An ODE-based Recursive Registration Network for Deformable Respiratory Motion Estimation With Lung 4DCT Images", IEEE TBME, 2023 (Liang *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/10144816)][[GitHub](https://github.com/lancial/orrn_public)]

### 2D-3D registration
* "The impact of machine learning on 2D/3D registration for image-guided interventions: A systematic review and perspective", FRAI, 2021 (Unberath *et al.*). [[Paper](https://www.frontiersin.org/articles/10.3389/frobt.2021.716007/full)]
* "Extended Capture Range of Rigid 2D/3D Registration by Estimating Riemannian Pose Gradients", MLMI, 2020 (Gu *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59861-7_29)]
* "Multiview 2D/3D rigid registration via a point-of-interest network for tracking and triangulation", CVPR, 2019 (Liao *et al.*). [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Liao_Multiview_2D3D_Rigid_Registration_via_a_Point-Of-Interest_Network_for_Tracking_CVPR_2019_paper.html)]
* "Generalizing spatial transformers to projective geometry with applications to 2D/3D registration", MICCAI, 2020 (Gao *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_32)][[GitHub](https://github.com/gaocong13/Projective-Spatial-Transformers)]
* "Fiducial-free 2D/3D registration of the proximal femur for robot-assisted femoroplasty", SPIE:MI, 2020 (Gao *et al.*). [[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11315/113151C/Fiducial-free-2D-3D-registration-of-the-proximal-femur-for/10.1117/12.2550992.short)]
* "Self-Supervised 2D/3D Registration for X-Ray to CT Image Fusion", WACV, 2019 (Jaganathan *et al.*). [[Paper](https://openaccess.thecvf.com/content/WACV2023/html/Jaganathan_Self-Supervised_2D3D_Registration_for_X-Ray_to_CT_Image_Fusion_WACV_2023_paper.html)]
* "A Novel Two-Stage Framework for 2D/3D Registration in Neurological Interventions", ROBIO, 2022 (Huang *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/10011812)]
* "X-ray to ct rigid registration using scene coordinate regression", MICCAI, 2023 (Shrestha *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74)][[GitHub](https://github.com/Pragyanstha/SCR-Registration)]
* "Extremely dense point correspondences using a learned feature descriptor", CVPR, 2020 (Liu *et al.*). [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Extremely_Dense_Point_Correspondences_Using_a_Learned_Feature_Descriptor_CVPR_2020_paper.html)][[GitHub](https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch)]
* "Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration", MedIA, 2023 (Bobrow *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841523002165?casa_token=m2rJHC2xQBUAAAAA:gbBK6cTlmKSIBWXUDSR8exzOlYWTgu4EfUtV-NrhK2dvZlyPkDMex4FR6LmVjiYAlDiYid4)][[GitHub](https://durrlab.github.io/C3VD/)]
* "StructuRegNet: Structure-Guided Multimodal 2D-3D Registration", MICCAI, 2023 (Leroy *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_73)]
* "A deep learning approach for 2D ultrasound and 3D CT/MR image registration in liver tumor ablation", CMPB, 2021 (Wei *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S0169260721001929?casa_token=kH3ZlrIdGIUAAAAA:OUA3CXZLlxM7g7-5iX7B7SVTaOCgXXM-SvPCl3UAgJBQ35kvp0h_oolb9U7t5kG7FBmUVzk)]
* "Multimodal registration of ultrasound and MR images using weighted self-similarity structure vector", CBM, 2023 (Wang *et al.*). [[Paper](https://www.sciencedirect.com/science/article/pii/S0010482523001269?casa_token=bLBV2JOg0BgAAAAA:pQOq04-a2cbcXi-Uxgd7KkjUnf1yVGuXCq34CXrTV-vRuGQTyhqoWYqX7ooevJZHyhB-MNM)]
* "Ultrasound Frame-to-Volume Registration via Deep Learning for Interventional Guidance", IEEE TUFFC, 2022 (Guo *et al.*). [[Paper](https://ieeexplore.ieee.org/document/9989409)][[GitHub](https://github.com/DIAL-RPI/Frame-to-Volume-Registration)]
* "A patient-specific self-supervised model for automatic X-Ray/CT registration", MICCAI, 2023 (Zhang *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_49)][[GitHub](https://github.com/BaochangZhang/PSSS_registration)]
* "X-ray to DRR images translation for efficient multiple objects similarity measures in deformable model 3D/2D registration", IEEE TMI, 2022 (Aubert *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9933846)]
* "Learning Expected Appearances for Intraoperative Registration during Neurosurgery", MICCAI, 2023 (Haouchine *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_22)][[GitHub](https://github.com/rouge1616/ExApp/)]
* "Non-Rigid 2D-3D Registration Using Convolutional Autoencoders", ISBI, 2020 (Li *et al.*). [[Paper](https://ieeexplore.ieee.org/document/9098602)]

## Towards Zero-shot Registration or Fundation Models
* "SynthMorph: learning contrast-invariant registration without acquired images", IEEE TMI, 2021 (Hoffmann *et al.*). [[Paper](https://ieeexplore.ieee.org/abstract/document/9552865)][[GitHub](https://martinos.org/malte/synthmorph/)]
* "Unsupervised 3D registration through optimization-guided cyclical self-training", MICCAI, 2023 (Bigalke *et al.*). [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_64)][[GitHub](https://github.com/multimodallearning/reg-cyclical-self-train)]
* "uniGradICON: A Foundation Model for Medical Image Registration", MICCAI, 2024 (Tian *et al.*). [[Paper](https://arxiv.org/abs/2403.05780)][[GitHub](https://github.com/uncbiag/uniGradICON)]
* "BrainMorph: A Foundational Keypoint Model for Robust and Flexible Brain MRI Registration", ArXiv, 2024 (Wang *et al.*). [[Paper](https://arxiv.org/abs/2405.14019v1)][[GitHub](https://github.com/alanqrwang/brainmorph)]
* $${\color{red}New!}$$ "multiGradICON: A Foundation Model for Multimodal Medical Image Registration", ArXiv, 2024 (Demir *et al.*). [[Paper](https://arxiv.org/abs/2408.00221)][[GitHub](https://github.com/uncbiag/uniGradICON)]

## Trends in image registration-related research based on PubMed paper counts
* Learning-based image registration research:
    * Search query: ```("image registration"[Title/Abstract] OR "image alignment"[Title/Abstract]) AND ("Neural Networks"[Title/Abstract] OR "Neural Network"[Title/Abstract] OR "DNN"[Title/Abstract] OR "CNN"[Title/Abstract] OR "ConvNet"[Title/Abstract] OR "Deep Learning"[Title/Abstract] OR "Transformer"[Title/Abstract])```
    * [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/?term=%28%22image+registration%22%5BTitle%2FAbstract%5D+OR+%22image+alignment%22%5BTitle%2FAbstract%5D%29+AND+%28%22Neural+Networks%22%5BTitle%2FAbstract%5D+OR+%22Neural+Network%22%5BTitle%2FAbstract%5D+OR+%22DNN%22%5BTitle%2FAbstract%5D+OR+%22CNN%22%5BTitle%2FAbstract%5D+OR+%22ConvNet%22%5BTitle%2FAbstract%5D+OR+%22Deep+Learning%22%5BTitle%2FAbstract%5D+OR+%22Transformer%22%5BTitle%2FAbstract%5D%29%0D%0A&sort=)
* Unsupervised learning-based image registration research:
    * Search query: ```("Unsupervised"[Title/Abstract] OR "end to end"[Title/Abstract]) AND ("Image Registration"[Title/Abstract] OR "Image Alignment"[Title/Abstract]) AND ("Neural Networks"[Title/Abstract] OR "Neural Network"[Title/Abstract] OR "DNN"[Title/Abstract] OR "CNN"[Title/Abstract] OR "ConvNet"[Title/Abstract] OR "Deep Learning"[Title/Abstract] OR "Transformer"[Title/Abstract])```
    * [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/?term=%28%22Unsupervised%22%5BTitle%2FAbstract%5D+OR+%22end+to+end%22%5BTitle%2FAbstract%5D%29+AND+%28%22Image+Registration%22%5BTitle%2FAbstract%5D+OR+%22Image+Alignment%22%5BTitle%2FAbstract%5D%29+AND+%28%22Neural+Networks%22%5BTitle%2FAbstract%5D+OR+%22Neural+Network%22%5BTitle%2FAbstract%5D+OR+%22DNN%22%5BTitle%2FAbstract%5D+OR+%22CNN%22%5BTitle%2FAbstract%5D+OR+%22ConvNet%22%5BTitle%2FAbstract%5D+OR+%22Deep+Learning%22%5BTitle%2FAbstract%5D+OR+%22Transformer%22%5BTitle%2FAbstract%5D%29%0D%0A&sort=)
