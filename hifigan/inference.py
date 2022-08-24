from env import AttrDict
from meldataset import MAX_WAV_VALUE, mel_spectrogram, load_wav
from models import Generator
import json
import torch

def load_hifigan(generator_path, config_path):
  with open(config_path) as f:
        data = f.read()
  json_config = json.loads(data)
  h = AttrDict(json_config)
  torch.manual_seed(h.seed)
  hifigan = Generator(h).to(torch.device('cuda'))
  state_dict_g = torch.load(generator_path, map_location=torch.device("cuda"))
  hifigan.load_state_dict(state_dict_g["generator"])
  hifigan.eval()
  hifigan.remove_weight_norm()
  return hifigan