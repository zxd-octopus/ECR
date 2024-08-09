import functools
import torch
import random
import os
import numpy as np


gpt2_special_tokens_dict = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<movie>'],
}

prompt_special_tokens_dict = {
    'additional_special_tokens': ['<movie>'],
}
Emo_List = ['like', 'curious', 'happy', 'grateful', 'negative', 'neutral', 'nostalgia', 'agreement', 'surprise']

Emo_loss_weight_dict = {'like':2.0, 'happy':2.0, 'curious':1.5, 'grateful':1.5, 'negative':1.0, 'neutral':0.5, 'nostalgia':1.5, 'agreement':1.5, 'surprise':1.5}
def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    print(f"Random seed set as {seed}")
    torch.set_deterministic(True)
