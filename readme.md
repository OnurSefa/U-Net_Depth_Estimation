# U-Net Depth Estimation

Goal of this project is gaining hands-on experience on monocular 
relative depth estimation task following [DenseDepth](https://arxiv.org/abs/1812.11941) paper.
The scripts and methodologies will imitate original paper using PyTorch.

Current results are shown in the below table. First column show the rgb input images, second one shows the 
ground truth depth values, and the third column shows the trained model's predictions. As it is visible from the
table, the model is able to capture generic monocular relative depth values with conserving the object structure
within the scene to some degree.

![milestone](https://github.com/OnurSefa/U-Net_Depth_Estimation/raw/main/docs/milestone.png)

The reasons why the training/fine-tuning operations stopped at this level:
1. The main goal of this project was to gain experience in depth estimation projects and understand the significant factors affecting the problem definition from the very start. And, this goal is achieved.
2. The project was not aimed to compete with SotA results. The model captures the monocular relative depth values and objects which is enough in this context.
3. The hardware requirements to fine-tune the model raised concerns about the limited resource, especially regarding time. There was no need to train it more.


## Update #2

### 2024.09.13

* Further fine-tuning operations completed

Below table shows the current model's predictions at the rightmost column. First column is the input rgb image, second column
shows the grand truth depth values. Third column shows the previous best model predictions.

![finetune2](https://github.com/OnurSefa/U-Net_Depth_Estimation/raw/main/docs/finetune2.png)


### 2024.09.08

The table shows the fine-tuned model results on the labeled sample.
Fine-tuning operation is held onto the previous step's best model.
The results are after a few experiments, further fine-tuning operations should be held.

![evaluate_finetune](https://github.com/OnurSefa/U-Net_Depth_Estimation/raw/main/docs/evaluate_finetune.png)

Below table shows comparison between fine-tuned model and previous model in the labeled sample dataset.

![final_comparison_fine_tune_data](https://github.com/OnurSefa/U-Net_Depth_Estimation/raw/main/docs/final_comparison_fine_tune_data.png)


Following table shows comparison between fine-tuned model and previous model in the raw dataset.

![final_comparison_raw_data](https://github.com/OnurSefa/U-Net_Depth_Estimation/raw/main/docs/final_comparison_raw_data.png)

* Initial fine-tune operation is handled, further training is required to get better results
* Documentation is enriched with image comparison tables 


### 2024.09.06

Following table shows the comparison between original-raw depth-predicted depth images.

![evalute](https://github.com/OnurSefa/U-Net_Depth_Estimation/raw/main/docs/evaluate.png)

The table shows how the raw dataset has missing points in some regions. Especially further space
is represented with all blacks which is not parallel with the definition (closer pixels should be darker).
The trained model captures blurry version of the goal.

* Retrieved data from [Original NYU Depth V2 Dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html#raw_parts)
* Selected a portion of the original dataset, because of the hardware and time constraints
* Tried to use inpainting methods, didn't get good results
* Added augmentation
  * Horizontal flip
  * Channel swap
* Trained the model on the selected portion of the dataset
* Evaluation with predicting original and mirrored image and taking the average

#### Next Steps
* Adding example images to the doc comparing current results - depth labels - original images 
* Fine-tuning the results with the fine-grained sample of the original dataset


## Update #1

### 2024.08.24 

Followed [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/pdf/1812.11941)
paper:
* Changed the encoder initiation from scratch to DensNet169. I took the pretrained
version of this model from the pytorch library.
* Started to clip the dataset images from all sides to eliminate the white spaces.
* Size of the depth prediction images is the half of the input images.
* Optimizer only trains the decoder part of the image. Encoder is frozen.

#### Next steps:
* Finding better NYU2 dataset version.
* Adding augmentation like mentioned in the paper.
* Inference with predicting original and mirrored image and taking the average.
* More experiments.

## Update #0

### 2024.08.18

Used [this](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py)
resource as a reference to implement the U-net model structure.

Used [NYU Depth V2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)
dataset that is stored in kaggle. I get rid of the test dataset because of the 
incompatibility between the test-train depth images. I split the training dataset
extracting the names of the test and train subgroups.

#### Issues:
* Trainings never converged.
* Some evaluated outputs showed all gray or noisy outputs. Most promising ones only learned how to extract edges or identify the color hues.
* Input dataset could be in better quality.
