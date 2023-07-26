from PIL import Image
import wandb
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        model = pl_module.model
        datalaoader = trainer.val_dataloaders[0]
        logger = pl_module.logger
        columns = ["image", "ground truth", "prediction"]

        for sample in datalaoader:
            images, ground_truth_masks = sample["image"], sample["label"]
            predicted_masks = model(images)
            break

        image_data = []

        for i, (image, mask, pred_mask) in enumerate(zip(images, ground_truth_masks, predicted_masks)):
            image_pil = image * 255
            image_pil = Image.fromarray(image_pil.detach().numpy()[0]).convert("L")
            mask_pil = mask * 255
            mask_pil = Image.fromarray(mask_pil.detach().numpy()[0]).convert("L")
            pred_mask_pil = pred_mask * 255
            pred_mask_pil = Image.fromarray(pred_mask_pil.detach().numpy()[0]).convert("L")
            image_data.append([wandb.Image(image_pil), wandb.Image(mask_pil), wandb.Image(pred_mask_pil)])

        logger.log_table(key='comparison', columns=columns, data=image_data)
