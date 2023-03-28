from omegaconf import OmegaConf
import datetime
import os
import torch
import sys
import shutil
from loguru import logger

####################
# Utils
####################
def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    res = model.load_state_dict(new_state_dict, strict=False)
    logger.info(res)
    return model

def load_conf(base_conf, include_exex_info=True, save_conf=True, save_code=True):
    cli_conf = OmegaConf.from_cli()
    if cli_conf.get('config',None) is not None:
        override_conf = OmegaConf.load(cli_conf.pop('config'))
    else:
        override_conf = OmegaConf.create()
    base_conf = OmegaConf.create(base_conf)
    conf =  OmegaConf.merge(base_conf, override_conf, cli_conf)

    if include_exex_info:
        exec_info = OmegaConf.create({'exec_info':{'script':sys.argv[0], 'time': str(datetime.datetime.today())}})
    conf = OmegaConf.merge(conf, exec_info)

    os.makedirs(conf.output_dir, exist_ok=True)
    if save_conf:
        OmegaConf.save(config=conf,f=os.path.join(conf.output_dir, 'config.yml'))
    if save_code:
        shutil.copy(sys.modules['__main__'].__file__, os.path.join(conf.output_dir, 'main.py'))

    return conf