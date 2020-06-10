import os
import glob
from pytorch_lightning.callbacks.base import Callback
from DIET.metrics import show_intent_report, show_entity_report
from DIET.dataset.intent_entity_dataset import RasaIntentEntityValidDataset

class PerfCallback(Callback):
    def __init__(self, file_path=None, gpu_num=0, report_nm=None, output_dir=None, root_path=None):
        self.file_path = file_path
        if gpu_num > 0:
            self.cuda = True
        else:
            self.cuda = False
        self.report_nm = report_nm
        self.output_dir = output_dir
        
        if root_path is None:
            self.root_path = 'lightning_logs'
        else:
            self.root_path = os.path.join(root_path, 'lightning_logs')

    def on_train_end(self, trainer, pl_module):        
        if self.file_path is None:
            print("evaluate valid data")
            dataset = pl_module.val_dataset
            tokenizer = pl_module.model.dataset.tokenizer
        else:
            print("evaluate new data")
            tokenizer = pl_module.model.dataset.tokenizer
            self.nlu_data = open(self.file_path, encoding="utf-8").readlines()
            dataset = RasaIntentEntityValidDataset(markdown_lines=self.nlu_data, tokenizer=tokenizer)
                
        if self.output_dir is None:
            folder_path = [f for f in glob.glob(os.path.join(self.root_path, "**/"), recursive=False)]
            folder_path.sort()
            self.output_dir  = folder_path[-1]
        self.output_dir = os.path.join(self.output_dir, 'results')
        intent_report_nm = self.report_nm.replace('.', '_intent.')
        entity_report_nm = self.report_nm.replace('.', '_entity.')
        show_intent_report(dataset, pl_module, tokenizer, file_name=intent_report_nm, output_dir=self.output_dir, cuda=self.cuda)
        show_entity_report(dataset, pl_module, file_name=entity_report_nm, output_dir=self.output_dir, cuda=self.cuda)
