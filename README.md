# 2D QPI segmentation pipeline

The segmentation pipeline consists of two deep learning models-

- Fine-tuned **Cellpose** — instance segmentation, identifying individual cell instances within a QPI MIP image.
- Custom **U-Net** — for multi-class semantic segmentation -  cell, nucleus, nucleolus, lipid droplets, and background

The two outputs are combined into a per-cell mask, which then feeds into downstream phenotyping (alignment and feature extraction on `FINAL_MIPs` / `FINAL_MASKS` - currently work in progress. The codes also use napari for 

# Repository structure

```
├── README.md                  
├── data_cleaning_pyfiles/     <- misc. data cleaning utilities
├── models/                    <- unet.py and unet_tests (model architecture)
├── qpi_seg/
│   ├── train/
│   │   ├── train_unet.py           <- train the U-Net
│   │   └── cellpose2d_train.py     <- fine-tune Cellpose
│   └── test/
│       ├── cellpose_test_napari_save.py  <- run Cellpose, edit in napari, save to CP_MASK/
│       ├── unet_test_save.py              <- run U-Net, save to UNET_MASK/
│       ├── save_combined_mask.py          <- combine per-cell masks into COMBINED_MASK/
│       └── run_napari_script.py           <- generic napari proofreading script
├── phenotyping/
│   └── align/                 <- alignment using FINAL_MIPs and FINAL_MASKS

├── environment.yml            <- conda environment spec
├── pyproject.toml
└── runs/Unet/                 <- training run logs and outputs
```
> **Note:** file ordering is not guaranteed by `os.listdir` — always use `sorted(os.listdir(...))` when matching MIPs to masks.

You'll need two saved models to run tests/inference:
1. **Cellpose (fine-tuned)** — [download link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/agupt130_jh_edu/IQDgbCtcT8siRI6GLLuhro0uASIsFZlhDyOwutltq0czLG4?e=ZDLTQj)
2. **U-Net** — [download link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/agupt130_jh_edu/IQCuq4fhppjxRbd9RnAlhxV_ARqKCvYyAgTsh4ZhQGkvFV4?e=fHOBhD)

Feel free to fine-tune using the train codes :)

## Setup 

Tested codes to work only on CPU cellpose+napari+torch all are in same environment) (for GPU install torch from official website according to CUDA version)

conda create -n artemis2d python=3.13
conda activate artemis2d
pip install torch torchvision   
pip install cellpose
python -m pip install "napari[all]"
```


## Acknowledgements

Some of the codes in this repository were developed with discussions with participants (especially members of tean ARTEMIS2D) of the ‘Deep Learning for Microscopy Image Analysis’ (DL@Janelia) at Howard Hughes Medical Institute (HHMI) Janelia Research Campus. U-Net implementation adapted from [dl-janelia/unet](https://github.com/dl-janelia/unet/tree/19d9ba70acf047ada35954144cabb78284bbbcde).

## License

Released under the [MIT License](LICENSE).
