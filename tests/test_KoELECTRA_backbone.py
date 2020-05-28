from unittest import TestCase

from DIET import trainer

import os, sys

#class Test_no_backbone_char_tokenizer(TestCase):
#    def test_train(self):
trainer.train(file_path="/home/ltech/.rasalt/data/nlu.md", backbone="koelectra")
