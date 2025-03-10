import argparse
import os.path as osp
import itertools
from omegaconf import OmegaConf
from semanticist.engine.trainer_utils import instantiate_from_config
from semanticist.utils.device_utils import configure_compute_backend

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Test a model")
    
    # Model and testing configuration
    parser.add_argument('--model', type=str, nargs='+', default=[None], help="Path to model directory")
    parser.add_argument('--step', type=int, nargs='+', default=[250000], help="Step number to test")
    parser.add_argument('--cfg', type=str, default=None, help="Path to config file")
    parser.add_argument('--dataset', type=str, default='imagenet', help="Dataset to use")

    # Legacy parameter (preserved for backward compatibility)
    parser.add_argument('--cfg_value', type=float, nargs='+', default=[None],
                       help='Legacy parameter for GPT classifier-free guidance scale')
    parser.add_argument('--ae_cfg', type=float, nargs='+', default=[None], 
                       help="Autoencoder classifier-free guidance scale")
    parser.add_argument('--cfg_schedule', type=str, nargs='+', default=[None], 
                       help="CFG schedule type (e.g., constant, linear)")
    parser.add_argument('--test_num_slots', type=int, nargs='+', default=[None], 
                       help="Number of slots to use for inference")
    parser.add_argument('--temperature', type=float, nargs='+', default=[None], 
                       help="Temperature for sampling")
    
    return parser.parse_args()


def load_config(model_path, cfg_path=None):
    """Load configuration from file or model directory."""
    if cfg_path is not None and osp.exists(cfg_path):
        config_path = cfg_path
    elif model_path and osp.exists(osp.join(model_path, 'config.yaml')):
        config_path = osp.join(model_path, 'config.yaml')
    else:
        raise ValueError(f"No config file found at {model_path} or {cfg_path}")
    
    return OmegaConf.load(config_path)


def setup_checkpoint_path(model_path, step, config):
    """Set up the checkpoint path based on model and step."""
    if model_path:
        ckpt_path = osp.join(model_path, 'models', f'step{step}')
        if not osp.exists(ckpt_path):
            print(f"Skipping non-existent checkpoint: {ckpt_path}")
            return None
        if hasattr(config.trainer.params, 'model'):
            config.trainer.params.model.params.ckpt_path = ckpt_path
        else:
            config.trainer.params.gpt_model.params.ckpt_path = ckpt_path
    else:
        result_folder = config.trainer.params.result_folder
        ckpt_path = osp.join(result_folder, 'models', f'step{step}')
        if hasattr(config.trainer.params, 'model'):
            config.trainer.params.model.params.ckpt_path = ckpt_path
        else:
            config.trainer.params.gpt_model.params.ckpt_path = ckpt_path
    
    return ckpt_path


def setup_test_config(config):
    """Set up common test configuration parameters."""
    config.trainer.params.test_dataset = config.trainer.params.dataset
    config.trainer.params.test_dataset.params.split = 'val'
    config.trainer.params.test_only = True
    config.trainer.params.compile = False
    config.trainer.params.eval_fid = True
    config.trainer.params.fid_stats = 'fid_stats/adm_in256_stats.npz'
    if hasattr(config.trainer.params, 'model'):
        config.trainer.params.model.params.num_sampling_steps = '250'
    else:
        config.trainer.params.ae_model.params.num_sampling_steps = '250'


def apply_cfg_params(config, param_dict):
    """Apply CFG-related parameters to the config."""
    # Apply each parameter if it's not None
    if param_dict.get('cfg_value') is not None:
        config.trainer.params.cfg = param_dict['cfg_value']
        print(f"Setting cfg to {param_dict['cfg_value']}")
    
    if param_dict.get('ae_cfg') is not None:
        config.trainer.params.ae_cfg = param_dict['ae_cfg']
        print(f"Setting ae_cfg to {param_dict['ae_cfg']}")

    if param_dict.get('cfg_schedule') is not None:
        config.trainer.params.cfg_schedule = param_dict['cfg_schedule']
        print(f"Setting cfg_schedule to {param_dict['cfg_schedule']}")

    if param_dict.get('test_num_slots') is not None:
        config.trainer.params.test_num_slots = param_dict['test_num_slots']
        print(f"Setting test_num_slots to {param_dict['test_num_slots']}")

    if param_dict.get('temperature') is not None:
        config.trainer.params.temperature = param_dict['temperature']
        print(f"Setting temperature to {param_dict['temperature']}")


def run_test(config):
    """Instantiate trainer and run test."""
    trainer = instantiate_from_config(config.trainer)
    trainer.train()


def generate_param_combinations(args):
    """Generate all combinations of parameters from the provided arguments."""
    # Create parameter grid for all combinations
    param_grid = {
        'cfg_value': [None] if args.cfg_value == [None] else args.cfg_value,
        'ae_cfg': [None] if args.ae_cfg == [None] else args.ae_cfg,
        'cfg_schedule': [None] if args.cfg_schedule == [None] else args.cfg_schedule,
        'test_num_slots': [None] if args.test_num_slots == [None] else args.test_num_slots,
        'temperature': [None] if args.temperature == [None] else args.temperature
    }
    
    # Get all parameter names that have non-None values
    active_params = [k for k, v in param_grid.items() if v != [None]]
    
    if not active_params:
        # If no parameters are specified, yield a dict with all None values
        yield {k: None for k in param_grid.keys()}
        return
    
    # Generate all combinations of active parameters
    active_values = [param_grid[k] for k in active_params]
    for combination in itertools.product(*active_values):
        param_dict = {k: None for k in param_grid.keys()}  # Start with all None
        for i, param_name in enumerate(active_params):
            param_dict[param_name] = combination[i]
        yield param_dict


def test(args):
    """Main test function that processes arguments and runs tests."""
    # Iterate through all model and step combinations
    for model in args.model:
        for step in args.step:
            print(f"Testing model: {model} at step: {step}")
            
            # Load configuration
            config = load_config(model, args.cfg)
            
            # Setup checkpoint path
            ckpt_path = setup_checkpoint_path(model, step, config)
            if ckpt_path is None:
                continue
            
            # Setup test configuration
            setup_test_config(config)
            
            # Generate and apply all parameter combinations
            for param_dict in generate_param_combinations(args):
                # Create a copy of the config for each parameter combination
                current_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                
                # Print parameter combination
                param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items() if v is not None])
                print(f"Testing with parameters: {param_str}")
                
                # Apply parameters and run test
                apply_cfg_params(current_config, param_dict)
                run_test(current_config)


def main():
    """Main entry point for the script."""
    args = parse_args()
    configure_compute_backend()
    test(args)


if __name__ == "__main__":
    main()
