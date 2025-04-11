# Kvasir-SEG Image Segmentation with U-Net

This repository implements an image segmentation model using the U-Net architecture to segment gastrointestinal images from the Kvasir-SEG dataset. The model is trained and evaluated using Dice score and Intersection over Union (IoU) as metrics.

## Project Structure

The project is organized into the following main directories and files:

- **`dataset/`**: Contains scripts to download and load the Kvasir-SEG dataset.
- **`model/`**: Contains the U-Net model definition.
- **`scripts/`**: Main training script and execution.
- **`utils/`**: Contains utility functions for metrics and visualization.
- **`.gitignore`**: Specifies which files and directories to ignore in the Git repository.
- **`README.md`**: Project documentation (this file).
- **`requirements.txt`**: Python dependencies for the project.

## Dataset

The Kvasir-SEG dataset can be downloaded from [here](https://datasets.simula.no/downloads/kvasir-seg.zip). Once downloaded, the dataset will be automatically extracted and used for training the segmentation model.

## Requirements

To run the project, you need to install the following dependencies:

- Python 3.x
- PyTorch
- OpenCV
- Albumentations
- tqdm
- scikit-learn
- matplotlib

To install the dependencies, run:

```bash
pip install -r requirements.txt


## Training the Model

To train the model, run the `train.py` script located in the `scripts/` directory. This will download the dataset (if not already downloaded), preprocess the data, and start training the U-Net model.

```bash
python scripts/train.py


The training process includes:

Loading and preprocessing images and masks.

Training the U-Net model on the dataset.

Saving the trained model to models/unet_kvasirseg.pth.

Model Architecture
The U-Net model architecture consists of the following components:

Encoder: A series of convolutional blocks that downsample the input image.

Bottleneck: The deepest part of the network where the most abstract features are learned.

Decoder: A series of transposed convolutions to upsample and recover the spatial resolution.

Skip Connections: These allow the network to directly pass feature maps from the encoder to the decoder.

Hyperparameters
Learning Rate: 1e-5

Batch Size: 8

Epochs: 100

Loss Function: A combination of binary cross-entropy and Dice coefficient.

Evaluation
During training, the model is evaluated on the validation set at the end of each epoch using the following metrics:

Dice Coefficient: A metric for the overlap between predicted and ground truth masks.

Intersection over Union (IoU): A metric to evaluate the quality of the predicted mask.

Validation metrics are printed at the end of each epoch.

Visualization
After every few epochs (every 5 epochs or at the final epoch), the following images will be displayed:

Input Image: The original input image.

Ground Truth Mask: The actual segmentation mask.

Predicted Mask: The mask predicted by the U-Net model.

These visualizations help track the progress of the model during training.

Saving the Model
Once training is complete, the trained model will be saved as a .pth file in the models/ directory:
```bash
models/unet_kvasirseg.pth


Acknowledgements
The Kvasir-SEG dataset used in this project is available from Simula Research Laboratory.

The U-Net model architecture was introduced by Olaf Ronneberger et al. in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015).