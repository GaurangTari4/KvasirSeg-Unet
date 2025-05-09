# 🚑 Kvasir-SEG Image Segmentation with U-Net

This repository implements an image segmentation model using the **U-Net** architecture to segment gastrointestinal images from the **Kvasir-SEG** dataset. The model is trained and evaluated using **Dice Score** and **Intersection over Union (IoU)** as performance metrics.

---

## 📁 Project Structure

```
KvasirSeg-Unet/
├── dataset/       # Dataset download and loading scripts
├── model/         # U-Net model definition
├── scripts/       # Training script
├── utils/         # Metrics and visualization utilities
├── models/        # Trained models
├── KvasirSeg-Unet_Demo.ipynb  # Colab demo notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Dataset

Download the Kvasir-SEG dataset from [here](https://datasets.simula.no/downloads/kvasir-seg.zip).

Once downloaded, the dataset will be automatically extracted and used for training and validation.

---

## ✅ Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

### Required Packages
- Python 3.x
- PyTorch
- OpenCV
- Albumentations
- tqdm
- scikit-learn
- matplotlib

---

## 🚀 Training the Model

To begin training the model:

```bash
python scripts/train.py
```

This script will:
- Download and prepare the dataset (if needed)
- Train the U-Net model on Kvasir-SEG
- Save the trained model to `models/unet_kvasirseg.pth`

---

## 🧠 U-Net Architecture

The U-Net model includes:
- **Encoder**: Downsampling path using convolutional layers
- **Bottleneck**: Deepest layer capturing abstract features
- **Decoder**: Upsampling path using transposed convolutions
- **Skip Connections**: Connect encoder features to corresponding decoder layers for better localization

---

## ⚙️ Hyperparameters

| Parameter       | Value        |
|----------------|--------------|
| Learning Rate  | 1e-5         |
| Batch Size     | 8            |
| Epochs         | 100          |
| Loss Function  | BCE + Dice   |

---

## 📊 Evaluation Metrics

- **Dice Coefficient**: Measures overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Evaluates segmentation quality

Metrics are printed after each training epoch.

---
## 🔗 Pretrained Model

The pre-trained U-Net model for Kvasir-SEG can be downloaded from the link below:

👉 [Download from Google Drive](https://drive.google.com/drive/folders/1lqBnKEcrdXOJnAPidseO2diU8faGdUK7?usp=sharing)

After downloading, place the `unet_kvasirseg.pth` file in the `models/` directory:

```
KvasirSeg-Unet/
├── models/
│   └── unet_kvasirseg.pth  ← Place it here
```

## 📒 Google Colab Demo

You can try the trained model **directly in your browser** via Google Colab without setting anything up locally:

👉 [**Run on Colab**](https://colab.research.google.com/github/GaurangTari4/KvasirSeg-Unet/blob/master/KvasirSeg_Unet_Demo.ipynb)

This notebook:
- Installs dependencies
- Downloads the Kvasir-SEG dataset
- Loads the pre-trained U-Net model
- Runs inference on sample images
- Visualizes the results

## 🖼️ Visualization

During training, visualizations are generated (every few epochs or final epoch), including:
- Input Image
- Ground Truth Mask
- Predicted Mask

This helps to monitor the model's learning progress.

---

## 💾 Saving the Model

The final trained model is saved to:

```
models/unet_kvasirseg.pth
```

You can later load this model for inference or fine-tuning.

---

## 🙏 Acknowledgements

- **Dataset**: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) by Simula Research Laboratory  
- **Model**: U-Net introduced in the paper _"U-Net: Convolutional Networks for Biomedical Image Segmentation"_ by Olaf Ronneberger et al. ([arXiv 2015](https://arxiv.org/abs/1505.04597))

---
