import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import timm
import lightning as L
from tqdm.auto import tqdm

# --- Hyperparameters ---
torch.set_float32_matmul_precision("high")
# Model
NUM_SLOTS = 128  # K: Number of slots to generate
TRANSFORMER_LAYERS = 3  # Number of processing layers in generator/reconstructor
MODELS = {
    "dino": "vit_base_patch16_224.dino",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",  # non reg version
    "dinov3": "vit_base_patch16_dinov3.lvd_1689m",
}
ENCODER_NAME = MODELS["dinov3"]
# Training
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
EPOCHS = 50
PRECISION = "bf16-mixed"
LOG_INTERVAL = 20
COMPILE = False


class SlotFormer(nn.Module):
    """
    Generates ordered, causal slots that explain vision transformer patch tokens.
    """

    def __init__(self, num_slots: int, num_layers: int, encoder_name: str):
        super().__init__()
        self.num_slots = num_slots
        
        # pretrained frozen ViT 
        self.encoder = timm.create_model(encoder_name, pretrained=True).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches

        # learnable queries that initiate the slot generation process.
        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.embed_dim))

        # positional embeddings for the decoder to reconstruct the patch grid.
        # initialized from the encoder's embeddings but made learnable.
        if self.encoder.pos_embed is not None:
            m = self.encoder.num_prefix_tokens
            pos_embed_init = self.encoder.pos_embed[:, m:].clone()
        else:
            # dinov3 uses ROPE, not 2d posembeds
            pos_embed_init = torch.empty(1, self.num_patches, self.embed_dim)
            nn.init.normal_(pos_embed_init, std=0.02)

        self.decoder_pos_embed = nn.Parameter(pos_embed_init)
        # autoregressive generator and reconstructor
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.generator = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Cache a causal target mask once for the generator (compile-friendly)
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(self.num_slots),
            persistent=False,
        )

        # Learnable NULL tokens for masked slots (per-slot)
        self.null_slots = nn.Parameter(torch.zeros(1, self.num_slots, self.embed_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.normal_(self.null_slots, std=0.02)

    def forward(self, x: Tensor):
        B, K = x.shape[0], self.num_slots
        # 1. get patch tokens Z
        with torch.no_grad():
            patch_tokens = self.encoder.forward_features(x)
            patch_tokens = patch_tokens[:, self.encoder.num_prefix_tokens :]  # drop cls

        # 2. generate slots S
        #   A single forward pass with a causal mask
        #   This ensures that generating s_i only depends on s_{1...i-1}.
        slots = self.generator(
            tgt=self.slot_queries.repeat(B, 1, 1),
            memory=patch_tokens,
            tgt_mask=self.tgt_mask.to(patch_tokens.device),
            tgt_is_causal=True,
        )  # [B, K, D]

        # 3. reconstruction
        # sample one prefix length m per sample and reconstruct once.
        arange = torch.arange(K, device=slots.device)
        m = torch.randint(1, K + 1, (B,), device=slots.device)
        keep = arange.unsqueeze(0) < m.unsqueeze(1)  # [B, K] bool

        # replace masked suffix with learnable NULL tokens
        keep_3d = keep.unsqueeze(-1)  # [B, K, 1] bool
        null = self.null_slots.expand(B, -1, -1).type_as(slots)  # [B, K, D]
        masked_slots = torch.where(keep_3d, slots, null)  # [B, K, D]

        reconstructed_patches = self.reconstructor(
            tgt=self.decoder_pos_embed.repeat(B, 1, 1),
            memory=masked_slots,
        )

        # 4. mse loss
        loss = F.mse_loss(reconstructed_patches, patch_tokens)
        return loss


def train():
    fabric = L.Fabric(accelerator="auto", precision=PRECISION)
    fabric.launch()

    model = SlotFormer(NUM_SLOTS, TRANSFORMER_LAYERS, ENCODER_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model = torch.compile(model, fullgraph=True, disable=not COMPILE)
    model, optimizer = fabric.setup(model, optimizer)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    data = torchvision.datasets.Imagenette(
        "/mnt/sdb1/datasets", split="train", transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    # --- Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch:02d}",
            leave=True,
        )
        for i, (images, _) in progress:
            optimizer.zero_grad()
            loss = model(images)
            fabric.backward(loss)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    fabric.print("Training finished")
    fabric.save(f"model-{epoch:02d}.ckpt", {"model": model})


if __name__ == "__main__":
    train()
