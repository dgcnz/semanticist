import os, torch
import os.path as osp
import shutil
import numpy as np
import copy
import torch.nn as nn
from tqdm.auto import tqdm
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, DistributedSampler
from semanticist.utils.logger import SmoothedValue, MetricLogger, empty_cache
from accelerate.utils import DistributedDataParallelKwargs
from semanticist.stage2.gpt import GPT_models
from semanticist.stage2.generate import generate
from pathlib import Path
import time

from semanticist.engine.trainer_utils import (
    instantiate_from_config, concat_all_gather,
    save_img_batch, get_fid_stats,
    EMAModel, create_scheduler, load_state_dict, load_safetensors,
    setup_result_folders, create_optimizer,
    CacheDataLoader
)

class GPTTrainer(nn.Module):
    def __init__(
        self,
        ae_model,
        gpt_model,
        dataset,
        test_only=False,
        num_test_images=50000,
        num_epoch=400,
        eval_classes=[1, 7, 282, 604, 724, 207, 250, 751, 404, 850], # goldfish, cock, tiger cat, hourglass, ship, golden retriever, husky, race car, airliner, teddy bear
        blr=1e-4,
        cosine_lr=False,
        lr_min=0,
        warmup_epochs=100,
        warmup_steps=None,
        warmup_lr_init=0,
        decay_steps=None,
        batch_size=32,
        cache_bs=8,
        test_bs=100,
        num_workers=8,
        pin_memory=False,
        max_grad_norm=None,
        grad_accum_steps=1,
        precision='bf16',
        save_every=10000,
        sample_every=1000,
        fid_every=50000,
        result_folder=None,
        log_dir="./log",
        ae_cfg=1.0,
        cfg=6.0,
        cfg_schedule="linear",
        temperature=1.0,
        train_num_slots=None,
        test_num_slots=None,
        eval_fid=False,
        fid_stats=None,
        enable_ema=False,
        compile=False,
        enable_cache_latents=True,
        cache_dir='/dev/shm/slot_cache'
    ):
        super().__init__()
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="tensorboard",
            project_dir=log_dir,
        )

        self.ae_model = instantiate_from_config(ae_model)
        ae_model_path = ae_model.params.ckpt_path
        assert ae_model_path.endswith(".safetensors") or ae_model_path.endswith(".pt") or ae_model_path.endswith(".pth") or ae_model_path.endswith(".pkl")
        assert osp.exists(ae_model_path), f"AE model checkpoint {ae_model_path} does not exist"
        self._load_checkpoint(ae_model_path, self.ae_model)

        self.ae_model.to(self.device)
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()

        self.model_name = gpt_model.target
        if 'GPT' in gpt_model.target:
            self.gpt_model = GPT_models[gpt_model.target](**gpt_model.params)
        else:
            raise ValueError(f"Unknown model type: {gpt_model.target}")
        self.num_slots = ae_model.params.num_slots
        self.slot_dim = ae_model.params.slot_dim

        self.test_only = test_only
        self.test_bs = test_bs
        self.num_test_images = num_test_images
        self.num_classes = gpt_model.params.num_classes
        self.batch_size = batch_size
        if not test_only:
            self.train_ds = instantiate_from_config(dataset)
            train_size = len(self.train_ds)
            if self.accelerator.is_main_process:
                print(f"train dataset size: {train_size}")

            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )
            self.train_dl = DataLoader(
                self.train_ds,
                batch_size=batch_size if not enable_cache_latents else cache_bs,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )

            effective_bs = batch_size * grad_accum_steps * self.accelerator.num_processes
            lr = blr * effective_bs / 256
            if self.accelerator.is_main_process:
                print(f"Effective batch size is {effective_bs}")

            self.g_optim = create_optimizer(self.gpt_model, weight_decay=0.05, learning_rate=lr)
            
            if warmup_epochs is not None:
                warmup_steps = warmup_epochs * len(self.train_dl)
                
            self.g_sched = create_scheduler(
                self.g_optim,
                num_epoch,
                len(self.train_dl),
                lr_min,
                warmup_steps,
                warmup_lr_init,
                decay_steps,
                cosine_lr
            )
            self.accelerator.register_for_checkpointing(self.g_sched)
            self.gpt_model, self.g_optim, self.g_sched = self.accelerator.prepare(self.gpt_model, self.g_optim, self.g_sched)
        else:
            self.gpt_model = self.accelerator.prepare(self.gpt_model)

        self.steps = 0
        self.loaded_steps = -1

        if compile:
            self.ae_model = torch.compile(self.ae_model, mode="reduce-overhead")
            _model = self.accelerator.unwrap_model(self.gpt_model)
            _model = torch.compile(_model, mode="reduce-overhead")

        self.enable_ema = enable_ema
        if self.enable_ema and not self.test_only: # when testing, we directly load the ema dict and skip here
            self.ema_model = EMAModel(self.accelerator.unwrap_model(self.gpt_model), self.device)
            self.accelerator.register_for_checkpointing(self.ema_model)

        self._load_checkpoint(gpt_model.params.ckpt_path)
        if self.test_only:
            self.steps = self.loaded_steps

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.sample_every = sample_every
        self.fid_every = fid_every
        self.max_grad_norm = max_grad_norm
        self.eval_classes = eval_classes
        self.cfg = cfg
        self.ae_cfg = ae_cfg
        self.cfg_schedule = cfg_schedule
        self.temperature = temperature
        self.train_num_slots = train_num_slots
        self.test_num_slots = test_num_slots
        if self.train_num_slots is not None:
            self.train_num_slots = min(self.train_num_slots, self.num_slots)
        else:
            self.train_num_slots = self.num_slots
        if self.test_num_slots is not None:
            self.num_slots_to_gen = min(self.test_num_slots, self.train_num_slots)
        else:
            self.num_slots_to_gen = self.train_num_slots
        self.eval_fid = eval_fid
        if eval_fid:
            assert fid_stats is not None
        self.fid_stats = fid_stats

        # Setup result folders
        self.result_folder = result_folder
        self.model_saved_dir, self.image_saved_dir = setup_result_folders(result_folder)

        # Setup cache
        self.cache_dir = Path(cache_dir)
        self.enable_cache_latents = enable_cache_latents
        self.cache_loader = None

    @property
    def device(self):
        return self.accelerator.device

    def _load_checkpoint(self, ckpt_path=None, model=None):
        if ckpt_path is None or not osp.exists(ckpt_path):
            return

        if model is None:
            model = self.accelerator.unwrap_model(self.gpt_model)

        if osp.isdir(ckpt_path):
            self.loaded_steps = int(
                ckpt_path.split("step")[-1].split("/")[0]
            )
            if not self.test_only:
                self.accelerator.load_state(ckpt_path)
            else:
                if self.enable_ema:
                    model_path = osp.join(ckpt_path, "custom_checkpoint_1.pkl")
                    if osp.exists(model_path):
                        state_dict = torch.load(model_path, map_location="cpu")
                        load_state_dict(state_dict, model)
                        if self.accelerator.is_main_process:
                            print(f"Loaded ema model from {model_path}")
                else:
                    model_path = osp.join(ckpt_path, "model.safetensors")
                    if osp.exists(model_path):
                        load_safetensors(model_path, model)
        else:
            if ckpt_path.endswith(".safetensors"):
                load_safetensors(ckpt_path, model)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                load_state_dict(state_dict, model)
        if self.accelerator.is_main_process:
            print(f"Loaded checkpoint from {ckpt_path}")

    def _build_cache(self):
        """Build cache for slots and targets."""
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        
        # Clean up any existing cache files first
        slots_file = self.cache_dir / f"slots_rank{rank}_of_{world_size}.mmap"
        targets_file = self.cache_dir / f"targets_rank{rank}_of_{world_size}.mmap"
        
        if slots_file.exists():
            os.remove(slots_file)
        if targets_file.exists():
            os.remove(targets_file)
        
        dataset_size = len(self.train_dl.dataset)
        shard_size = dataset_size // world_size
        
        # Detect number of augmentations from first batch
        with torch.no_grad():
            sample_batch = next(iter(self.train_dl))
            img, _ = sample_batch
            num_augs = img.shape[1] if len(img.shape) == 5 else 1
        
        print(f"Rank {rank}: Creating new cache with {num_augs} augmentations per image...")
        os.makedirs(self.cache_dir, exist_ok=True)
        slots_file = self.cache_dir / f"slots_rank{rank}_of_{world_size}.mmap"
        targets_file = self.cache_dir / f"targets_rank{rank}_of_{world_size}.mmap"
        
        # Create memory-mapped files
        slots_mmap = np.memmap(
            slots_file,
            dtype='float32',
            mode='w+',
            shape=(shard_size * num_augs, self.train_num_slots, self.slot_dim)
        )
        
        targets_mmap = np.memmap(
            targets_file,
            dtype='int64',
            mode='w+',
            shape=(shard_size * num_augs,)
        )
        
        # Cache data
        with torch.no_grad():
            for i, batch in enumerate(tqdm(
                self.train_dl, 
                desc=f"Rank {rank}: Caching data",
                disable=not self.accelerator.is_local_main_process
            )):
                imgs, targets = batch
                if len(imgs.shape) == 5:  # [B, num_augs, C, H, W]
                    B, A, C, H, W = imgs.shape
                    imgs = imgs.view(-1, C, H, W)  # [B*num_augs, C, H, W]
                    targets = targets.unsqueeze(1).expand(-1, A).reshape(-1)  # [B*num_augs]
                
                # Split imgs into n chunks
                num_splits = num_augs
                split_size = imgs.shape[0] // num_splits
                imgs_splits = torch.split(imgs, split_size)
                targets_splits = torch.split(targets, split_size)
                
                start_idx = i * self.train_dl.batch_size * num_augs
                
                for split_idx, (img_split, targets_split) in enumerate(zip(imgs_splits, targets_splits)):
                    img_split = img_split.to(self.device, non_blocking=True)
                    slots_split = self.ae_model.encode_slots(img_split)[:, :self.train_num_slots, :]
                    
                    split_start = start_idx + (split_idx * split_size)
                    split_end = split_start + img_split.shape[0]
                    
                    # Write directly to mmap files
                    slots_mmap[split_start:split_end] = slots_split.cpu().numpy()
                    targets_mmap[split_start:split_end] = targets_split.numpy()
        
        # Close the mmap files
        del slots_mmap
        del targets_mmap
        
        # Reopen in read mode
        self.cached_latents = np.memmap(
            slots_file,
            dtype='float32',
            mode='r',
            shape=(shard_size * num_augs, self.train_num_slots, self.slot_dim)
        )
        
        self.cached_targets = np.memmap(
            targets_file,
            dtype='int64',
            mode='r',
            shape=(shard_size * num_augs,)
        )
        
        # Store the number of augmentations for the cache loader
        self.num_augs = num_augs

    def _setup_cache(self):
        """Setup cache if enabled."""
        self._build_cache()
        self.accelerator.wait_for_everyone()

        # Initialize cache loader if cache exists
        if self.cached_latents is not None:
            self.cache_loader = CacheDataLoader(
                slots=self.cached_latents,
                targets=self.cached_targets,
                batch_size=self.batch_size,
                num_augs=self.num_augs,
                seed=42 + self.accelerator.process_index
            )

    def __del__(self):
        """Cleanup cache files."""
        if self.enable_cache_latents:
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            
            # Clean up slots cache
            slots_file = self.cache_dir / f"slots_rank{rank}_of_{world_size}.mmap"
            if slots_file.exists():
                os.remove(slots_file)
            
            # Clean up targets cache
            targets_file = self.cache_dir / f"targets_rank{rank}_of_{world_size}.mmap"
            if targets_file.exists():
                os.remove(targets_file)

    def _train_step(self, slots, targets=None):
        """Execute single training step."""
        
        with self.accelerator.accumulate(self.gpt_model):
            with self.accelerator.autocast():
                loss = self.gpt_model(slots, targets)
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.gpt_model.parameters(), self.max_grad_norm)
            self.g_optim.step()
            if self.g_sched is not None:
                self.g_sched.step_update(self.steps)
            self.g_optim.zero_grad()

        # Update EMA model if enabled
        if self.enable_ema:
            self.ema_model.update(self.accelerator.unwrap_model(self.gpt_model))
        
        return loss

    def _train_epoch_cached(self, epoch, logger):
        """Train one epoch using cached data."""
        self.cache_loader.set_epoch(epoch)
        header = f'Epoch: [{epoch}/{self.num_epoch}]'
        
        for batch in logger.log_every(self.cache_loader, 20, header):
            slots, targets = (b.to(self.device, non_blocking=True) for b in batch)
            
            self.steps += 1

            if self.steps == 1:
                print(f"Training batch size: {len(slots)}")
                print(f"Hello from index {self.accelerator.local_process_index}")
            
            loss = self._train_step(slots, targets)
            self._handle_periodic_ops(loss, logger)

    def _train_epoch_uncached(self, epoch, logger):
        """Train one epoch using raw data."""
        header = f'Epoch: [{epoch}/{self.num_epoch}]'
        
        for batch in logger.log_every(self.train_dl, 20, header):
            img, targets = (b.to(self.device, non_blocking=True) for b in batch)
            
            self.steps += 1
            
            if self.steps == 1:
                print(f"Training batch size: {img.size(0)}")
                print(f"Hello from index {self.accelerator.local_process_index}")

            slots = self.ae_model.encode_slots(img)[:, :self.train_num_slots, :]
            loss = self._train_step(slots, targets)
            self._handle_periodic_ops(loss, logger)

    def _handle_periodic_ops(self, loss, logger):
        """Handle periodic operations and logging."""
        logger.update(loss=loss.item())
        logger.update(lr=self.g_optim.param_groups[0]["lr"])
        
        if self.steps % self.save_every == 0:
            self.save()
        
        if (self.steps % self.sample_every == 0) or (self.eval_fid and self.steps % self.fid_every == 0):
            empty_cache()
            self.evaluate()
            self.accelerator.wait_for_everyone()
            empty_cache()

    def _save_config(self, config):
        """Save configuration file."""
        if config is not None and self.accelerator.is_main_process:
            import shutil
            from omegaconf import OmegaConf

            if isinstance(config, str) and osp.exists(config):
                shutil.copy(config, osp.join(self.result_folder, "config.yaml"))
            else:
                config_save_path = osp.join(self.result_folder, "config.yaml")
                OmegaConf.save(config, config_save_path)

    def _should_skip_epoch(self, epoch):
        """Check if epoch should be skipped due to loaded checkpoint."""
        loader = self.train_dl if not self.enable_cache_latents else self.cache_loader
        if ((epoch + 1) * len(loader)) <= self.loaded_steps:
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch} is skipped because it is loaded from ckpt")
            self.steps += len(loader)
            return True
        
        if self.steps < self.loaded_steps:
            for _ in loader:
                self.steps += 1
                if self.steps >= self.loaded_steps:
                    break
        return False

    def train(self, config=None):
        """Main training loop."""
        # Initial setup
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            print(f"number of learnable parameters: {n_parameters//1e6}M")
        
        self._save_config(config)
        self.accelerator.init_trackers("gpt")

        # Handle test-only mode
        if self.test_only:
            empty_cache()
            self.evaluate()
            self.accelerator.wait_for_everyone()
            empty_cache()
            return

        # Setup cache if enabled
        if self.enable_cache_latents:
            self._setup_cache()

        # Training loop
        for epoch in range(self.num_epoch):
            if self._should_skip_epoch(epoch):
                continue

            self.gpt_model.train()
            logger = MetricLogger(delimiter="  ")
            logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

            # Choose training path based on cache availability
            if self.enable_cache_latents:
                self._train_epoch_cached(epoch, logger)
            else:
                self._train_epoch_uncached(epoch, logger)

            # Synchronize and log epoch stats
            logger.synchronize_between_processes()
            if self.accelerator.is_main_process:
                print("Averaged stats:", logger)

        # Finish training
        self.accelerator.end_training()
        self.save()
        if self.accelerator.is_main_process:
            print("Train finished!")

    def save(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(
            os.path.join(self.model_saved_dir, f"step{self.steps}")
        )

    @torch.no_grad()
    def evaluate(self, use_ema=True):
        self.gpt_model.eval()
        unwraped_gpt_model = self.accelerator.unwrap_model(self.gpt_model)
        # switch to ema params, only when eval_fid is True
        # if test_only, we directly load the ema dict and skip here
        use_ema = use_ema and self.enable_ema and self.eval_fid and not self.test_only
        if use_ema:
            if hasattr(self, "ema_model"):
                model_without_ddp = self.accelerator.unwrap_model(self.gpt_model)
                model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
                ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
                for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
                    if "nested_sampler" in name:
                        continue
                    ema_state_dict[name] = self.ema_model.state_dict()[name]
                if self.accelerator.is_main_process:
                    print("Switch to ema")
                model_without_ddp.load_state_dict(ema_state_dict)
            else:
                print("EMA model not found, using original model")
                use_ema = False

        if not self.test_only:
            classes = torch.tensor(self.eval_classes, device=self.device)
            with self.accelerator.autocast():
                slots = generate(unwraped_gpt_model, classes, self.num_slots_to_gen, cfg_scale=self.cfg, cfg_schedule=self.cfg_schedule, temperature=self.temperature)
                if self.num_slots_to_gen < self.num_slots:
                    null_slots = self.ae_model.dit.null_cond.expand(slots.shape[0], -1, -1)
                    null_slots = null_slots[:, self.num_slots_to_gen:, :]
                    slots = torch.cat([slots, null_slots], dim=1)
                imgs = self.ae_model.sample(slots, targets=classes, cfg=self.ae_cfg) # targets are not used for now

            imgs = concat_all_gather(imgs)
            if self.accelerator.num_processes > 16:
                imgs = imgs[:16*len(self.eval_classes)]
            imgs = imgs.detach().cpu()
            grid = make_grid(
                imgs, nrow=len(self.eval_classes), normalize=True, value_range=(0, 1)
            )
            if self.accelerator.is_main_process:
                save_image(
                    grid,
                    os.path.join(
                        self.image_saved_dir, f"step{self.steps}_aecfg-{self.ae_cfg}_cfg-{self.cfg_schedule}-{self.cfg}_slots{self.num_slots_to_gen}_temp{self.temperature}.jpg"
                    ),
                )
        if self.eval_fid and (self.test_only or (self.steps % self.fid_every == 0)):
            # Create output directory (only on main process)
            save_folder = os.path.join(self.image_saved_dir, f"gen_step{self.steps}_aecfg-{self.ae_cfg}_cfg-{self.cfg_schedule}-{self.cfg}_slots{self.num_slots_to_gen}_temp{self.temperature}")
            if self.accelerator.is_main_process:
                os.makedirs(save_folder, exist_ok=True)

            # Setup for distributed generation
            world_size = self.accelerator.num_processes
            local_rank = self.accelerator.process_index
            batch_size = self.test_bs
            
            # Create balanced class distribution
            num_classes = self.num_classes
            images_per_class = self.num_test_images // num_classes
            class_labels = np.repeat(np.arange(num_classes), images_per_class)
            
            # Shuffle the class labels to ensure random ordering
            np.random.shuffle(class_labels)
            
            total_images = len(class_labels)

            padding_size = world_size * batch_size - (total_images % (world_size * batch_size))
            class_labels = np.pad(class_labels, (0, padding_size), 'constant')
            padded_total_images = len(class_labels)

            # Distribute workload across GPUs
            images_per_gpu = padded_total_images // world_size
            start_idx = local_rank * images_per_gpu
            end_idx = min(start_idx + images_per_gpu, padded_total_images)
            local_class_labels = class_labels[start_idx:end_idx]
            local_num_steps = len(local_class_labels) // batch_size
            
            if self.accelerator.is_main_process:
                print(f"Generating {total_images} images ({images_per_class} per class) across {world_size} GPUs")
            
            used_time = 0
            gen_img_cnt = 0
            
            for i in range(local_num_steps):
                if self.accelerator.is_main_process and i % 10 == 0:
                    print(f"Generation step {i}/{local_num_steps}")

                # Get and pad labels for current batch
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                labels = local_class_labels[batch_start:batch_end]
                
                # Convert to tensors and track real vs padding
                labels = torch.tensor(labels, device=self.device)
                
                # Generate images
                self.accelerator.wait_for_everyone()
                start_time = time.time()
                with torch.no_grad():
                    with self.accelerator.autocast():
                        slots = generate(unwraped_gpt_model, labels, self.num_slots_to_gen, 
                                            cfg_scale=self.cfg, 
                                            cfg_schedule=self.cfg_schedule,
                                            temperature=self.temperature)
                        if self.num_slots_to_gen < self.num_slots:
                            null_slots = self.ae_model.dit.null_cond.expand(slots.shape[0], -1, -1)
                            null_slots = null_slots[:, self.num_slots_to_gen:, :]
                            slots = torch.cat([slots, null_slots], dim=1)
                        imgs = self.ae_model.sample(slots, targets=labels, cfg=self.ae_cfg)

                samples_in_batch = min(batch_size * world_size, total_images - gen_img_cnt)

                # Update timing stats
                used_time += time.time() - start_time
                gen_img_cnt += samples_in_batch
                if self.accelerator.is_main_process and i % 10 == 0:
                    print(f"Avg generation time: {used_time/gen_img_cnt:.5f} sec/image")

                gathered_imgs = concat_all_gather(imgs)
                gathered_imgs = gathered_imgs[:samples_in_batch]
                
                # Save images (only on main process)
                if self.accelerator.is_main_process:
                    real_imgs = gathered_imgs.detach().cpu()
                    
                    save_paths = [
                        os.path.join(save_folder, f"{str(idx).zfill(5)}.png")
                        for idx in range(gen_img_cnt - samples_in_batch, gen_img_cnt)
                    ]
                    save_img_batch(real_imgs, save_paths)

            # Calculate metrics (only on main process)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                generated_files = len(os.listdir(save_folder))
                print(f"Generated {generated_files} images out of {total_images} expected")
                
                metrics_dict = get_fid_stats(save_folder, None, self.fid_stats)
                fid = metrics_dict["frechet_inception_distance"]
                inception_score = metrics_dict["inception_score_mean"]
                
                metric_prefix = "fid_ema" if use_ema else "fid"
                isc_prefix = "isc_ema" if use_ema else "isc"
                
                self.accelerator.log({
                    metric_prefix: fid,
                    isc_prefix: inception_score,
                    "gpt_cfg": self.cfg,
                    "ae_cfg": self.ae_cfg,
                    "cfg_schedule": self.cfg_schedule,
                    "temperature": self.temperature,
                    "num_slots": self.test_num_slots if self.test_num_slots is not None else self.train_num_slots
                }, step=self.steps)
                
                # Print comprehensive CFG information
                cfg_info = (
                    f"{'EMA ' if use_ema else ''}CFG params: "
                    f"gpt_cfg={self.cfg}, ae_cfg={self.ae_cfg}, "
                    f"cfg_schedule={self.cfg_schedule}, "
                    f"num_slots={self.test_num_slots if self.test_num_slots is not None else self.train_num_slots}, "
                    f"temperature={self.temperature}"
                )
                print(cfg_info)
                print(f"FID: {fid:.2f}, ISC: {inception_score:.2f}")

                # Cleanup
                shutil.rmtree(save_folder)

        # back to no ema
        if use_ema:
            if self.accelerator.is_main_process:
                print("Switch back from ema")
            model_without_ddp.load_state_dict(model_state_dict)

        self.gpt_model.train()

