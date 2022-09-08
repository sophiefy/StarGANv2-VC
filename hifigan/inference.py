from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath)
    print("Complete.")
    return checkpoint_dict
  
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def load_model(model_path, config_path):
  with open(config_path) as f:
    data = f.read()
    
  json_config = json.loads(data)
  h = AttrDict(json_config)
  torch.manual_seed(h.seed)
  generator = Generator(h)
  state_dict_g = load_checkpoint(model_path)
  generator.load_state_dict(state_dict_g['generator'])
  
  return generator
