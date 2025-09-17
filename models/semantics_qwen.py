"""
Qwen2.5-VL-3B-Instruct Semantics Encoder
Wraps Qwen2.5-VL-3B-Instruct to produce multimodal embeddings s ∈ ℝ^{B×Seq_len×3584}.
Trains the entire Qwen backbone end-to-end for optimal performance.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLModel
from qwen_vl_utils import process_vision_info


class QwenSemanticsEncoder(nn.Module):
    """
    Wraps Qwen2.5-VL-3B-Instruct to produce multimodal embeddings s ∈ ℝ^{B×Seq_len×3584}.
    Trains the entire Qwen backbone end-to-end for optimal performance.
    
    Fixed model: Qwen/Qwen2.5-VL-3B-Instruct
    Output dimensions: [Batch_size, Sequence_length, 3584]
    """
    def __init__(
        self,
        min_pixels=None,
        max_pixels=None,
        device_map="auto",
        dtype=torch.bfloat16
    ):
        super().__init__()
        
        # Fixed model: Qwen2.5-VL-3B-Instruct
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
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
        

        # Enable gradient checkpointing for memory efficiency
        self.backbone.gradient_checkpointing_enable()
        
        # Keep backbone trainable for end-to-end learning
        
        # Store hidden size for reference
        # Qwen2.5-VL-3B-Instruct has hidden_size = 3584
        self.hidden_size = self.backbone.config.hidden_size  # 3584

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
            s: Semantic embedding [B, Seq_len, 3584]
            - B: Batch size
            - Seq_len: Variable sequence length (image + text tokens)
            - 3584: Fixed hidden dimension for Qwen2.5-VL-3B-Instruct
        """
        # Pack inputs (always allow gradients since backbone is trainable)
        inputs = self._pack(images, texts)
        
        # Move to correct device (use backbone parameters since we don't have proj layer yet)
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get hidden states from backbone
        out = self.backbone(
            **inputs,
            output_hidden_states=False,
            return_dict=True
        )
        
        out = out.last_hidden_state  # Shape: [B, Seq_len, 3584]
        
        # Output the last hidden state, no projection
        # Final output shape: [B, Seq_len, 3584] for Qwen2.5-VL-3B-Instruct
        return out