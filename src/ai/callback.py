import os

from PIL import Image
import wandb
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        model = pl_module.model
        datalaoader = trainer.val_dataloaders[0]
        for sample in datalaoader:
            images, ground_truth_masks = sample["image"], sample["label"]
            predicted_masks = model(images)
            break

        for i, (image, mask, pred_mask) in enumerate(zip(images, ground_truth_masks, predicted_masks)):
            image_pil = Image.fromarray(image.detach().numpy()[0]).convert("L")
            mask_pil = Image.fromarray(mask.detach().numpy()[0]).convert("L")
            pred_mask_pil = Image.fromarray(pred_mask.detach().numpy()[0]).convert("L")

            masked_image = wandb.Image(
            image_pil,
            masks={
            "predictions": {"mask_data": pred_mask_pil},
            "ground_truth": {"mask_data": mask_pil},
            },
            )

            pl_module.log({"img_with_masks": masked_image})
