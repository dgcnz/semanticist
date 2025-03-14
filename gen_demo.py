import gradio as gr
import numpy as np
from PIL import Image
import os.path as osp
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from semanticist.engine.trainer_utils import instantiate_from_config
from semanticist.stage1.diffuse_slot import DiffuseSlot
from semanticist.stage2.gpt import GPT_models
from semanticist.stage2.generate import generate
from safetensors import safe_open
from semanticist.utils.datasets import vae_transforms
from PIL import Image
from imagenet_classes import imagenet_classes

transform = vae_transforms('test')


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))

from PIL import Image
def convert_np(img):
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255)\
            .permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr
def convert_PIL(img):
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255)\
            .permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    return img

def norm_slots(slots):
    mean = torch.mean(slots, dim=-1, keepdim=True)
    std = torch.std(slots, dim=-1, keepdim=True)
    return (slots - mean) / std

def load_state_dict(state_dict, model):
    """Helper to load a state dict with proper prefix handling."""
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # Remove '_orig_mod' prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False
    )
    # print(f"Loaded model. Missing: {missing}, Unexpected: {unexpected}")

def load_safetensors(path, model):
    """Helper to load a safetensors checkpoint."""
    from safetensors.torch import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    load_state_dict(state_dict, model)

def load_checkpoint(ckpt_path, model):
    if ckpt_path is None or not osp.exists(ckpt_path):
        return

    if osp.isdir(ckpt_path):
        # ckpt_path is something like 'path/to/models/step10/'
        model_path = osp.join(ckpt_path, "model.safetensors")
        if osp.exists(model_path):
            load_safetensors(model_path, model)
    else:
        # ckpt_path is something like 'path/to/models/step10.pt'
        if ckpt_path.endswith(".safetensors"):
            load_safetensors(ckpt_path, model)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            load_state_dict(state_dict, model)

    print(f"Loaded checkpoint from {ckpt_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Is CUDA available: {torch.cuda.is_available()}")
if device == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

ckpt_path = hf_hub_download(repo_id='tennant/semanticist', filename="semanticist_ar_gen_L.pkl", cache_dir='/mnt/ceph_rbd/mnt_pvc_vid_data/zbc/cache/')
config_path = 'configs/autoregressive_xl.yaml'

cfg = OmegaConf.load(config_path)
params = cfg.trainer.params

ae_model = instantiate_from_config(params.ae_model).to(device)
ae_model_path = hf_hub_download(repo_id='tennant/semanticist', filename="semanticist_tok_XL.pkl", cache_dir='/mnt/ceph_rbd/mnt_pvc_vid_data/zbc/cache/')
load_checkpoint(ae_model_path, ae_model)
ae_model.eval()

gpt_model = GPT_models[params.gpt_model.target](**params.gpt_model.params).to(device)
load_checkpoint(ckpt_path, gpt_model)
gpt_model.eval();

def viz_diff_slots(model, slots, nums, cfg=1.0, return_figs=False):
    n_slots_inf = []
    for num_slots_to_inference in nums:
        drop_mask = model.nested_sampler(slots.shape[0], device, num_slots_to_inference)
        recon_n = model.sample(slots, drop_mask=drop_mask, cfg=cfg)
        n_slots_inf.append(recon_n)
    return [convert_np(n_slots_inf[i][0]) for i in range(len(n_slots_inf))]

num_slots = params.ae_model.params.num_slots
slot_dim = params.ae_model.params.slot_dim
dtype = torch.bfloat16
# the model is trained with only 32 tokens.
num_slots_to_gen = 32

# Function to generate image from class
def generate_from_class(class_id, cfg_scale):
    with torch.no_grad():
        dtype = torch.bfloat16
        num_slots_to_gen = 32
        with torch.autocast(device, dtype=dtype):
            slots_gen = generate(
                gpt_model, 
                torch.tensor([class_id]).to(device), 
                num_slots_to_gen, 
                cfg_scale=cfg_scale, 
                cfg_schedule="linear"
            )
        if num_slots_to_gen < num_slots:
            null_slots = ae_model.dit.null_cond.expand(slots_gen.shape[0], -1, -1)
            null_slots = null_slots[:, num_slots_to_gen:, :]
            slots_gen = torch.cat([slots_gen, null_slots], dim=1)
    return slots_gen

with gr.Blocks() as demo:
    with gr.Row():
        # First column - Input and configs
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            
            # Replace image input with ImageNet class selection
            imagenet_classes = {k: v for k, v in enumerate(imagenet_classes)}
            class_choices = [f"{id}: {name}" for id, name in imagenet_classes.items()]
            
            # Dropdown for class selection
            class_dropdown = gr.Dropdown(
                choices=class_choices[:20],  # Limit for demonstration
                label="Select ImageNet Class",
                value=class_choices[0] if class_choices else None
            )
            
            # Option to enter class ID directly
            class_id_input = gr.Number(
                label="Or enter class ID directly (0-999)",
                value=0,
                minimum=0,
                maximum=999,
                step=1
            )
            
            with gr.Group():
                gr.Markdown("### Configuration")
                show_gallery = gr.Checkbox(label="Show Gallery", value=True)
                slider = gr.Slider(minimum=0.1, maximum=20.0, value=4.0, label="CFG value")
                labels_input = gr.Textbox(
                    label="Number of tokens to reconstruct (comma-separated)", 
                    value="1, 2, 4, 8, 16", 
                    placeholder="Enter comma-separated numbers for the number of slots to use"
                )
        
        # Second column - Output (conditionally rendered)
        with gr.Column(scale=1):
            gr.Markdown("## Output")
            
            # Container for conditional rendering
            with gr.Group(visible=True) as gallery_container:
                gallery = gr.Gallery(label="Result Gallery", columns=3, height="auto", show_label=True)
            
            # Always visible output image
            output_image = gr.Image(label="Generated Image", type="numpy")
    
    # Handle form submission
    submit_btn = gr.Button("Generate")
    
    # Define the processing logic
    def update_outputs(class_selection, class_id, show_gallery_value, slider_value, labels_text):
        # Determine which class to use - either from dropdown or direct input
        if class_selection:
            # Extract class ID from the dropdown selection
            selected_class_id = int(class_selection.split(":")[0])
        else:
            selected_class_id = int(class_id)
        
        # Update the visibility of the gallery container
        gallery_container.visible = show_gallery_value
        
        try:
            # Parse the labels from the text input
            if labels_text and "," in labels_text:
                labels = [int(label.strip()) for label in labels_text.split(",")]
            else:
                # Default labels if none provided or in wrong format
                labels = [1, 4, 16, 64, 256]
        except:
            labels = [1, 4, 16, 64, 256]
        
        while len(labels) < 3:
            labels.append(256)
        
        # Generate the image based on the selected class
        slots_gen = generate_from_class(selected_class_id, cfg_scale=slider_value)
        
        recon = viz_diff_slots(ae_model, slots_gen, [32], cfg=slider_value)[0]
        
        # Always generate the model decomposition for potential gallery display
        model_decompose = viz_diff_slots(ae_model, slots_gen, labels, cfg=slider_value)
        
        if not show_gallery_value:
            # If only the image should be shown, return just the processed image
            return gallery_container, [], recon
        else:
            # Create image variations and pair them with labels
            gallery_images = [
                (recon, f'Generated from class {selected_class_id}'),
            ] + [(img, 'Gen. with ' + str(label) + ' tokens') for img, label in zip(model_decompose, labels)]
            return gallery_container, gallery_images, recon
    
    # Connect the inputs and outputs
    submit_btn.click(
        fn=update_outputs,
        inputs=[class_dropdown, class_id_input, show_gallery, slider, labels_input],
        outputs=[gallery_container, gallery, output_image]
    )
    
    # Also update when checkbox changes
    show_gallery.change(
        fn=lambda value: gr.update(visible=value),
        inputs=[show_gallery],
        outputs=[gallery_container]
    )

    # Add examples
    examples = [
        # ["0: tench, Tinca tinca", 0, True, 4.0, "1,2,4,8,16"],
        ["1: goldfish", 1, True, 4.0, "1,2,4,8,16"],
        # ["2: great white shark, white shark", 2, True, 4.0, "1,2,4,8,16"],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[class_dropdown, class_id_input, show_gallery, slider, labels_input],
        outputs=[gallery_container, gallery, output_image],
        fn=update_outputs,
        cache_examples=False
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()
