# U-Net Depth Estimation


## 2024.08.28 - Update #2

* Retrieved data from [Original NYU Depth V2 Dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html#raw_parts)
* Selected a portion of it, because of the hardware and time considerations
* Tried to use inpainting methods, didn't get good results
  * Will train the model on the large dataset, then fine tune it on the curated dataset sample
* Added augmentation
  * Horizontal flip
  * Channel swap
* Inference with predicting original and mirrored image and taking the average

### Current Plans
* Making experiments with the prepared dataset sample (bigger than the previous dataset)
* Fine-tuning the results with the fine-grained sample of the original dataset


## 2024.08.24 - Update #1

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
* Inference with predicting original and mirrored image and taking the average.
* More experiments.

## 2024.08.18 - Update #0
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
