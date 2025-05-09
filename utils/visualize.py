import matplotlib.pyplot as plt
import torch

def plot_prediction(inputs, masks, preds):
    inputs = inputs.cpu().permute(0, 2, 3, 1).detach().numpy()
    masks = masks.cpu().squeeze(1).detach().numpy()
    preds = torch.sigmoid(preds).cpu().squeeze(1).detach().numpy()

    batch_size = inputs.shape[0]
    fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axs = [axs]

    for i in range(batch_size):
        axs[i][0].imshow(inputs[i])
        axs[i][0].set_title("Input")
        axs[i][1].imshow(masks[i], cmap="gray")
        axs[i][1].set_title("Ground Truth")
        axs[i][2].imshow(preds[i], cmap="gray")
        axs[i][2].set_title("Prediction")

        for ax in axs[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
