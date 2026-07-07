About qpi_seg folder:

Cellpose is used for instance segmentation to identify different cell instances in a QPI MIP image. UNET is used to segment cell, nucleus, nucleolus, lipid droplet and background (5 channels) from QPI image

Predict scripts - 
For inferring masks from Unet: 'predict_unseen.py'. For inference from cellpose (afterfinetuning): 'cellpose_pretrained_eval.py', for inference from original cellpose: 'cellpose_eval.py'.

To use the predict_unseen.py file you would need the saved model that is https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/agupt130_jh_edu/IQCuq4fhppjxRbd9RnAlhxV_ARqKCvYyAgTsh4ZhQGkvFV4?e=fHOBhD

Instructions to start it working on PC (use VS studio terminal)
1. git clone https://github.com/anoshqa/ARTEMIS2D
2. ls artemis2d (to see files inside the folder)
3. cd artemis2d (goes to artemis2d folder)
once you are on the folder 
4. conda create -n artemis2d (suggested name of new environment)
5. conda install conda-forge::python==3.13.0
6. pip3 install torch torchvision (for downloading torch on Windows without GPU)
7. pip install cellpose

You will also need two saved models 
1. cellpose pre-trained model (trained on 700 breast cancer MIPs so far)
2. UNet 

