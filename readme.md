# U-Net Depth Estimation


I used [this](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py)
resource as a reference to implement the U-net model structure.

I used [NYU Depth V2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)
dataset that is stored in kaggle. I get rid of the test dataset because of the 
incompatibility between the test-train depth images. I split the training dataset
extracting the names of the test and train subgroups.

