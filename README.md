UNET is used to segment cell, nucleus, nucleolus, lipid droplet and background (5 channels) from QPI images of sizes greater than (832, 832). Cellpose is used for instance segmentation.

Please put new image folder path in 'predict_unseen.py' script inside qpi_seg folder to test out masks from Unet. To test out masks from cellpose (afterfinetuning), use the 'cellpose_pretrained_eval.py', for original cellpose use 'cellpose_eval.py'.

To use the predict_unseen.py file you would need the saved model that is https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/agupt130_jh_edu/IQCuq4fhppjxRbd9RnAlhxV_ARqKCvYyAgTsh4ZhQGkvFV4?e=fHOBhD

