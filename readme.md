# U-Net Depth Estimation


## 2024.08.24 -> Update #1

Followed [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/pdf/1812.11941)
paper:
* Changed the encoder initiation from scratch to DensNet169. I took the pretrained
version of this model from the pytorch library.
* Started to clip the dataset images from all sides to eliminate the white spaces.
* Size of the depth prediction images is the half of the input images.
* Optimizer only trains the decoder part of the image. Encoder is frozen.

### Next steps:
* Finding better NYU2 dataset version.
* Adding augmentation like mentioned in the paper.
* More experiments.

## 2024.08.18 -> Update #0
Used [this](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py)
resource as a reference to implement the U-net model structure.

Used [NYU Depth V2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)
dataset that is stored in kaggle. I get rid of the test dataset because of the 
incompatibility between the test-train depth images. I split the training dataset
extracting the names of the test and train subgroups.

### Issues:
* Trainings never converged.
* Some evaluated outputs showed all gray or noisy outputs. Most promising ones only learned how to extract edges or identify the color hues.
* Input dataset could be in better quality.
