from pytorch_lightning import Trainer

from .DIET_lightning_model import DualIntentEntityTransformer
from .mnist_model import MNISTModel

import os, sys


def train(file_path, checkpoint_path=os.getcwd(), max_epochs=10):
    trainer = Trainer(
        default_save_path=checkpoint_path,
        max_epochs=max_epochs,
    )
    model = DualIntentEntityTransformer(data_file_path=file_path)
    #model = MNISTModel()

    trainer.fit(model)
