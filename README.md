# A survey on deep learning in medical image registration: new technologies, uncertainty, evaluation metrics, and beyond
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2307.15615-b31b1b.svg)](https://arxiv.org/abs/2307.15615)

This official repository contains Python implementations of various image similarity measures, deformation regularization techniques, and evaluation methodologies for medical image registration.


## Image similarity measure:
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

## Deformation regularization:
1. Diffusion regularization
2. Total-variation regularization
3. Bending energy


## Citations:
These Python implementations have been prepared for inclusion in the following article:

    @article{chen2023survey,
    title={A survey on deep learning in medical image registration: New technologies, uncertainty, evaluation metrics, and beyond},
    author={Chen, Junyu and Liu, Yihao and Wei, Shuwen and Bian, Zhangxing and Subramanian, Shalini and Carass, Aaron and Prince, Jerry L and Du, Yong},
    journal={arXiv preprint arXiv:2307.15615},
    year={2023}
    }


