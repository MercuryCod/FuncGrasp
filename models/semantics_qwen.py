"""
Qwen2.5-VL Semantics Encoder
Wraps Qwen2.5-VL to produce a pooled multimodal embedding s ∈ ℝ^{B×Csem_proj}.
Default: freeze Qwen and learn only a projection.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLModel
from qwen_vl_utils import process_vision_info


class QwenSemanticsEncoder(nn.Module):
    """
    Wraps Qwen2.5-VL to produce a pooled multimodal embedding s ∈ ℝ^{B×Csem_proj}.
    Default: freeze Qwen and learn only a projection.
    """
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        csem_proj=256,
        freeze_backbone=True,  # Control whether to freeze Qwen
        min_pixels=None,
        max_pixels=None,
        device_map="auto",
        dtype=torch.bfloat16
    ):
        super().__init__()
        
        # Initialize processor for handling images and text
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # Use the *backbone* (no LM head) to access hidden states
        self.backbone = Qwen2_5_VLModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map
        )
        
        # Optionally freeze Qwen parameters
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        else:
            # Enable gradient checkpointing for memory efficiency
            self.backbone.gradient_checkpointing_enable()
        
        # Learnable projection layer
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, csem_proj)
        )

    def _pack(self, images, texts):
        """
        Pack images and texts into processor format.
        
        Args:
            images: list of PIL.Image or paths
            texts: list[str], length B
        
        Returns:
            Processed inputs ready for the model
        """
        # One-message-per-sample batching
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": tx}
                ]
            }
            for im, tx in zip(images, texts)
        ]
        
        # Apply chat template
        text_inputs = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages
        ]
        
        # Process vision info
        image_inputs, video_inputs = zip(*[process_vision_info(m) for m in messages])
        
        # Create processor inputs
        proc = self.processor(
            text=text_inputs,
            images=list(image_inputs),
            videos=list(video_inputs),
            padding=True,
            return_tensors="pt"
        )
        
        return proc

    def forward(self, images, texts):
        """
        Forward pass to get semantic embedding.
        
        Args:
            images: List[PIL.Image]
            texts: List[str]
        
        Returns:
            s: Semantic embedding [B, Csem_proj]
        """
        # Pack inputs (with or without gradients based on freeze_backbone)
        if self.backbone.training and any(p.requires_grad for p in self.backbone.parameters()):
            inputs = self._pack(images, texts)
        else:
            with torch.inference_mode():
                inputs = self._pack(images, texts)
        
        # Move to correct device
        device = next(self.proj.parameters()).device
        inputs = inputs.to(device)
        
        # Get hidden states from backbone
        out = self.backbone(
            **inputs,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Masked mean pool over tokens (text+vision) → global multimodal vector
        mask = inputs.attention_mask.float().unsqueeze(-1)  # [B, L, 1]
        pooled = (out.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1e-6))
        
        # Project to semantic embedding space
        s = self.proj(pooled)  # [B, Csem_proj]
        
        return s