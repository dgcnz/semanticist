import gradio as gr
import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from semanticist.engine.trainer_utils import instantiate_from_config
from semanticist.stage1.diffuse_slot import DiffuseSlot

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = hf_hub_download(repo_id='tennant/semanticist', filename='semanticist_tok_XL.pkl', cache_dir='/mnt/ceph_rbd/mnt_pvc_vid_data/zbc/cache/')
config_path = 'configs/tokenizer_xl.yaml'
cfg = OmegaConf.load(config_path)
ckpt = torch.load(ckpt_path, map_location='cpu')
from semanticist.utils.datasets import vae_transforms
from PIL import Image

transform = vae_transforms('test')
print(f"Is CUDA available: {torch.cuda.is_available()}")
if device == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


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

ckpt = {k.replace('._orig_mod', ''): v for k, v in ckpt.items()}

model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])
msg = model.load_state_dict(ckpt, strict=False)
model = model.to(device)
model = model.eval()
model.enable_nest = True

def viz_diff_slots(model, img, nums, cfg=1.0, return_img=False):
    n_slots_inf = []
    for num_slots_to_inference in nums:
        recon_n = model(
            img, sample=True, cfg=cfg,
            inference_with_n_slots=num_slots_to_inference,
        )
        n_slots_inf.append(recon_n)
    return [convert_np(n_slots_inf[i][0]) for i in range(len(n_slots_inf))]

# Removed process_image function as its functionality is now in the update_outputs function

with gr.Blocks() as demo:
    with gr.Row():
        # First column - Input and configs
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            input_image = gr.Image(label="Upload an image", type="numpy")
            
            with gr.Group():
                gr.Markdown("### Configuration")
                show_gallery = gr.Checkbox(label="Show Gallery", value=True)
                # You can add more config options here
                # slider = gr.Slider(minimum=0, maximum=10, value=5, label="Processing Intensity")
                slider = gr.Slider(minimum=0.1, maximum=20.0, value=4.0, label="CFG value")
                labels_input = gr.Textbox(
                    label="Number of tokens to reconstruct (comma-separated)", 
                    value="1, 4, 16, 64, 256", 
                    placeholder="Enter comma-separated numbers for the number of slots to use"
                )
        
        # Second column - Output (conditionally rendered)
        with gr.Column(scale=1):
            gr.Markdown("## Output")
            
            # Container for conditional rendering
            with gr.Group(visible=True) as gallery_container:
                gallery = gr.Gallery(label="Result Gallery", columns=3, height="auto", show_label=True)
            
            # Always visible output image
            output_image = gr.Image(label="Processed Image", type="numpy")
    
    # Handle form submission
    submit_btn = gr.Button("Process")
    
    # Define the processing logic
    def update_outputs(image, show_gallery_value, slider_value, labels_text):
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
        
        # Process the image based on configurations
        if image is None:
            # Return placeholder if no image is uploaded
            placeholder = np.zeros((300, 300, 3), dtype=np.uint8)
            return gallery_container, [], placeholder
        image = Image.fromarray(image)
        img = transform(image)
        img = img.unsqueeze(0).to(device)
        recon = viz_diff_slots(model, img, [256], cfg=slider_value)[0]

            
        if not show_gallery_value:
            # If only the image should be shown, return just the processed image
            return gallery_container, [], recon
        else:
            model_decompose = viz_diff_slots(model, img, labels, cfg=slider_value)
            # Create image variations and pair them with labels
            gallery_images = [
                (image, 'GT'),
                # (np.array(Image.fromarray(image).convert("L").convert("RGB")), labels[1]),
                # (np.array(Image.fromarray(image).rotate(180)), labels[2])
            ] + [(img, 'Recon. with ' + str(label) + ' tokens') for img, label in zip(model_decompose, labels)]
            return gallery_container, gallery_images, image
    
    # Connect the inputs and outputs
    submit_btn.click(
        fn=update_outputs,
        inputs=[input_image, show_gallery, slider, labels_input],
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
        ["examples/city.jpg", True, 4.0, "1,4,16,64,256"],
        ["examples/food.jpg", True, 4.0, "1,4,16,64,256"],
        ["examples/highland.webp", True, 4.0, "1,4,16,64,256"],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[input_image, show_gallery, slider, labels_input],
        outputs=[gallery_container, gallery, output_image],
        fn=update_outputs,
        cache_examples=True
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()
