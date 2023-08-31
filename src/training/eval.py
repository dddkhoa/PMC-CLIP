from copy import deepcopy

import clip
import torch
import torch.optim as optim

from training.params import parse_args
from pmc_clip.factory import create_model_and_transforms, _MODEL_CONFIGS
from pmc_clip.model import PMC_CLIP


class TempArgsConfig:
    def __init__(self):
        self.model = 'RN50_fusion4'
        self.precision = 'amp'
        self.force_quick_gelu = False
        self.pretrained_image = None
        self.pretrained = ''
        # self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.mlm = False
        self.crop_scale = 0.5
        self.wd = 0.2



def eval():
    args = TempArgsConfig()

    model_name = args.model
    model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    bert_model_name = model_cfg['text_cfg']['bert_model_name']


    PATH = "../../checkpoint.pt" # TODO: change path to checkpoint

    model, preprocess_train, preprocess_val = create_model_and_transforms(
            args=args,
            precision=args.precision,
            device=args.device,
            jit=None,
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image,
        )
    
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
            lr=5.0e-4,
            betas=(0.9, 0.999),
            eps=1.0e-8,
    )
    
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


eval()