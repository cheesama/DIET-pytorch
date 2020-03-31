from pytorch_lightning import Trainer

from .DIET_lightning_model import DualIntentEntityTransformer


def train(file_path):
    trainer = Trainer()
    model = DualIntentEntityTransformer(data_file_path=file_path)

    trainer.fit(model)
