import matplotlib.pyplot as plt
import torch

def plot_prediction(inputs, masks, preds):
    inputs = inputs.permute(0, 2, 3, 1).numpy()
    masks = masks.squeeze(1).numpy()
    preds = (preds.squeeze(1).numpy() > 0.5).astype(int)

    num_samples = min(3, len(inputs))
    for i in range(num_samples):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(inputs[i])
        ax[0].set_title("Input Image")
        ax[1].imshow(masks[i], cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(preds[i], cmap="gray")
        ax[2].set_title("Prediction")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()
