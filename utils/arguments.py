import argparse
import os
import shutil
from pathlib import Path
import yaml


class Configs:
    def __init__(self):
        self.REQUIRE_CONFGS = None
        self.cfgs = self.__ADD_CONFIGS_PARAMS()
        # assert self.REQUIRE_CONFGS is not None
        self.__INITIALIZE()


    def __ADD_CONFIGS_PARAMS(self):
        parser = argparse.ArgumentParser()

        # self.REQUIRE_CONFGS = ['epoch', 'batch_size', 'learning_rate', 'weight_decay']

        # If config file directory is provided, then load config from file.
        parser.add_argument('-cf', '--config_dir', 
                            type=str, 
                            default=None, 
                            help='If config file directory is provided, then load config from file.')

        # global setting
        parser.add_argument('--inferencing', 
                            default=False, 
                            action='store_true', 
                            help='When this arugument is provided, the running mode will be set to inference mode')
        parser.add_argument('--dataset', 
                            type=str, 
                            default='GRAB', 
                            help='Selected dataset [default: GRAB]')
        parser.add_argument('--dataset_dir', 
                            type=str, 
                            default=None, 
                            help='Selected dataset directory')

        # Data setting
        parser.add_argument('--max_frame', 
                            type=int, 
                            default=150,
                            help='Maximum motion length [default: 150]')

        # model setting
        ## mano
        parser.add_argument('--mano_model_path', 
                            type=str, 
                            default=None, 
                            help='Pre-trained MANO model path')
        ## Transformer
        parser.add_argument('--n_head', 
                            type=int, 
                            default=4,
                            help='N-head self-attention in Transformer Encoder [default: 4]')
        parser.add_argument('--n_layers', 
                            type=int, 
                            default=8,
                            help='N-layers Transformer Encoder [default: 8]')
        ## Contact map generation
        parser.add_argument('--fps_npoint', 
                            type=int, default=1024,
                            help='N-point cloud sampled by farthest point sampling (FPS) algorithm [default: 1024]')
        parser.add_argument('--second_stn', 
                            default=False, action='store_true',
                            help='When this arugument is provided, PointNet adopt the second STN (Spatial Transform Network)')
        ## Motion generator
        parser.add_argument('--beta_schedule', 
                            type=str, 
                            default='linear', 
                            help='Beta scheduler of DDPM [default: linear]')
        parser.add_argument('--timesteps', 
                            type=int, 
                            default=1000,
                            help='DDPM timesteps [default: 1000]')
        ## Hand refiner
        parser.add_argument('--contect_loss_lambda', 
                            type=int, 
                            default=5,
                            help='Contact loss lambda [default: 1000]')
        
        # training setting
        parser.add_argument('-cp', '--checkpoint_dir', 
                            type=str, default='checkpoints/tmp', 
                            help='Model checkpoint directory [default: checkpoints/tmp]')
        parser.add_argument('--resume', 
                            default=False, action='store_true',
                            help='When this arugument is provided, the program will resume from checkpoint file (only in training mode)')
        parser.add_argument('--epoch', 
                            type=int, default=5, 
                            help='Epoch to run [default: 5]')
        parser.add_argument('--batch_size', 
                            type=int, default=4, 
                            help='Batch Size during training [default: 16]')
        parser.add_argument('--learning_rate', 
                            type=float, default=0.001, 
                            help='Initial learning rate [default: 0.001]')
        parser.add_argument('--weight_decay', 
                            type=float, default=0, 
                            help='Optimization L2 weight decay [default: 0]')
        
        ## Loss
        parser.add_argument('--tau', 
                            type=float, default=2, 
                            help='hand-object distance threshold [default: 2]')
        
        
        # testing setting
        parser.add_argument('-rd', '--result_path', 
                            type=str, default='results/tmp', 
                            help='Outputs save path [default: results/tmp]')
        
        return parser.parse_args()
    
    def __INITIALIZE(self):
        if not self.cfgs.inferencing and self.cfgs.resume and self.cfgs.config_dir is None:
            raise ValueError("There are three ways to run the porject:\n" + \
                "- 1. Training with <--config_dir> when not resuming from checkpoint." + \
                "- 2. Training with <--resume> [--checkpoint_dir] when resuming from checkpoint." + \
                "- 3. Inferencing with <--inference> [--checkpoint_dir] [--result_path].")

        if self.cfgs.inferencing and self.cfgs.resume:
            raise ValueError("Inference and resuming from a checkpoint cannot take place simultaneously.")
        if self.cfgs.resume and self.cfgs.config_dir is not None:
            raise ValueError("Resuming from a checkpoint and manually loading configuration cannot take place simultaneously.")
        if self.cfgs.inferencing and self.cfgs.config_dir is not None:
            raise ValueError("Inference do not require manually loading configuration.")
        
        if self.cfgs.inferencing:
            result_path = self.cfgs.__dict__["result_path"]
            self.__LOAD_CHECKPOINT_CONFIGS()
            self.cfgs.__dict__["inferencing"] = True
            self.cfgs.__dict__["result_path"] = result_path
            return

        if self.cfgs.resume:
            self.__LOAD_CHECKPOINT_CONFIGS()
            self.cfgs.__dict__['resume'] = True
            return
        
        if self.cfgs.config_dir is None:
            raise ValueError("Training requires manually loading configuration with --config_dir when not resuming from checkpoint.")
        
        self.__CLEAR_CHECKPOINT_PATH_FILE()
        self.__LOAD_CONFIGS(self.cfgs.config_dir)
        self.__SAVE_CONFIGS()

        # raise ValueError("There are three ways to run the porject:\n" + \
        #     "- 1. Training with <--config_dir> when not resuming from checkpoint." + \
        #     "- 2. Training with <--resume> [--checkpoint_dir] when resuming from checkpoint." + \
        #     "- 3. Inferencing with <--inference> [--checkpoint_dir] [--result_path].")

    def __LOAD_CONFIGS(self, dir):
        with open(dir, 'r') as yaml_file:
            loaded_cfgs = yaml.safe_load(yaml_file)

        for key, value in loaded_cfgs.items():
            if key in self.cfgs.__dict__:
                self.cfgs.__dict__[key] = value

    def __LOAD_CHECKPOINT_CONFIGS(self):
        self.__LOAD_CONFIGS(f'{self.cfgs.checkpoint_dir}/configs.yml')
        
    def __CLEAR_CHECKPOINT_PATH_FILE(self):
        for elm in Path(self.cfgs.checkpoint_dir).glob('*'):
            elm.unlink() if elm.is_file() else shutil.rmtree(elm)

    def __SAVE_CONFIGS(self):
        if not os.path.exists(self.cfgs.checkpoint_dir):
            os.makedirs(self.cfgs.checkpoint_dir)
        cfgs_dict = self.cfgs.__dict__
        # filtered_cfgs_dict = {k: v for k, v in cfgs_dict.items() if k in self.REQUIRE_CONFGS}

        with open(f'{self.cfgs.checkpoint_dir}/configs.yml', 'w') as yaml_file:
            yaml.dump(cfgs_dict, yaml_file, default_flow_style=False)

CFGS = Configs().cfgs
pass
