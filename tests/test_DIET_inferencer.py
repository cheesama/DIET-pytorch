from DIET import Inferencer

import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="checkpoint_path")
parser.add_argument("text", help="inference target text")
args = parser.parse_args()

inferencer = Inferencer(args.checkpoint)
pp = pprint.PrettyPrinter(indent=4)
print ('\n infer result: ')
pp.pprint (inferencer.inference(args.text))
