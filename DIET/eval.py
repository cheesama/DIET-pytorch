import os
import glob
from pytorch_lightning.callbacks.base import Callback
from DIET.metrics import show_intent_report, show_entity_report
from DIET.dataset.intent_entity_dataset import RasaIntentEntityDataset

class PerfCallback(Callback):
    def __init__(self, file_path=None, gpu_num=0, report_nm=None, output_dir=None):
        self.file_path = file_path
        if gpu_num > 0:
            self.cuda = True
        else:
            self.cuda = False
        self.report_nm = report_nm
        self.output_dir = output_dir

    def on_train_end(self, trainer, pl_module):
        print("train finished")
        if self.file_path is None:
            dataset = pl_module.val_dataset
        else:
            dataset = RasaIntentEntityDataset(markdown_lines=self.file_path, tokenizer=None)
                
        if self.output_dir is None:
            path = 'lightning_logs/'
            folder_path = [f for f in glob.glob(path + "**/", recursive=False)]
            folder_path.sort()
            self.output_dir  = folder_path[-1]
        self.output_dir = os.path.join(self.output_dir, 'results')
        intent_report_nm = self.report_nm.replace('.', '_intent.')
        show_intent_report(dataset, pl_module, file_name=intent_report_nm, output_dir=self.output_dir, cuda=self.cuda)
