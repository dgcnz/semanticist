# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import submitit

from omegaconf import OmegaConf
from semanticist.engine.trainer_utils import instantiate_from_config
from semanticist.utils.device_utils import configure_compute_backend

def parse_args():
    parser = argparse.ArgumentParser("Submitit for accelerator training")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=7000, type=int, help="Duration of the job, default 5 days")
    parser.add_argument("--qos", default="normal", type=str, help="QOS to request")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="your-partition", type=str, help="Partition where to submit")
    parser.add_argument("--exclude", default="", type=str, help="Exclude nodes from the partition")
    parser.add_argument("--nodelist", default="", type=str, help="Nodelist to request")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument('--cfg', type=str, default='configs/your_config.yaml', help='accelerator configs')
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def __call__(self):
        self._setup_gpu_args()
        configure_compute_backend()
        trainer = instantiate_from_config(self.config.trainer)
        trainer.train(self.config)

    def checkpoint(self):
        import os
        import submitit

        model_dir = os.path.join(self.args.output_dir, "models")
        if os.path.exists(model_dir):
            # Get all step folders
            step_folders = [d for d in os.listdir(model_dir) if d.startswith("step")]
            if step_folders:
                # Extract step numbers and find max
                steps = [int(f.replace("step", "")) for f in step_folders]
                max_step = max(steps)
                # Set ckpt path to the latest step folder
                self.config.trainer.params.model.params.ckpt_path = os.path.join(model_dir, f"step{max_step}")
        print("Requeuing ", self.args, self.config)
        empty_trainer = type(self)(self.args, self.config)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")

        job_env = submitit.JobEnvironment()
        self.args.output_dir = str(self.args.output_dir).replace("%j", str(job_env.job_id))
        self.args.log_dir = self.args.output_dir
        self.config.trainer.params.result_folder = self.args.output_dir
        self.config.trainer.params.log_dir = os.path.join(self.args.output_dir, "logs")
        # self.args.gpu = job_env.local_rank
        # self.args.rank = job_env.global_rank
        # self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    cfg_file = args.cfg
    assert os.path.exists(cfg_file)
    config = OmegaConf.load(cfg_file)

    if config.trainer.params.result_folder is None:
        if args.job_dir == "":
            args.job_dir = "./output/%j"
        
        config.trainer.params.result_folder = args.job_dir
        config.trainer.params.log_dir = os.path.join(args.job_dir, "logs")
    else:
        args.job_dir = config.trainer.params.result_folder

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    qos = args.qos

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment
    if args.exclude:
        kwargs["slurm_exclude"] = args.exclude
    if args.nodelist:
        kwargs["slurm_nodelist"] = args.nodelist

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        # cpus_per_task=16,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_qos=qos,
        **kwargs
    )

    executor.update_parameters(name="semanticist")

    args.output_dir = args.job_dir

    trainer = Trainer(args, config)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
