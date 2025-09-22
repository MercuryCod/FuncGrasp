"""
Qwen2.5-VL-3B-Instruct Semantics Encoder
Wraps Qwen2.5-VL-3B-Instruct and returns batched token hidden states H ∈ ℝ^{B×L_max×2048}
for downstream masked pooling. Trains the entire Qwen backbone end-to-end.
Each sample in the batch has its own text and set of images.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class QwenSemanticsEncoder(nn.Module):
    """
    Wraps Qwen2.5-VL-3B-Instruct to produce batched token hidden states H ∈ ℝ^{B×L_max×2048}.
    Trains the entire Qwen backbone end-to-end for optimal performance.
    
    Fixed model: Qwen/Qwen2.5-VL-3B-Instruct
    Output: last_hidden_state [B, L_max, 2048] where B is batch size
    Each sample in the batch has its own text prompt and set of images.
    """
    def __init__(
        self,
        min_pixels=None,
        max_pixels=None,
        device=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        
        # Fixed model: Qwen2.5-VL-3B-Instruct
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_name = os.path.join(MODEL_PATH, model_name)
        
        # Initialize processor for handling images and text
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # Use the *backbone* (no LM head) to access hidden states
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None  # Don't use device_map to avoid multi-GPU issues
        )
        
        # Move to device if specified
        if device is not None:
            self.backbone = self.backbone.to(device)
        

        # Enable gradient checkpointing for memory efficiency
        self.backbone.gradient_checkpointing_enable()
        
        # Keep backbone trainable for end-to-end learning
        
        # Store hidden size for reference
        # Qwen2.5-VL-3B-Instruct has hidden_size = 2048
        self.hidden_size = self.backbone.config.hidden_size  # 2048

    def _pack_batch(self, images_list, texts_list):
        """
        Pack batched images and texts into processor format.
        
        Args:
            images_list: List[List[PIL.Image]] - B lists of images
            texts_list: List[str] - B text prompts
        
        Returns:
            Processed inputs ready for the model
        """
        all_messages = []
        all_text_inputs = []
        
        for images, text in zip(images_list, texts_list):
            # Format messages for each sample
            if images:
                content = []
                for img in images:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": text})
            else:
                content = [{"type": "text", "text": text}]
            
            messages = [{"role": "user", "content": content}]
            all_messages.append(messages)
            
            # Apply chat template
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_text_inputs.append(text_input)
            
        # Process vision info once for the whole batch (aligned with official docs)
        image_inputs, video_inputs = process_vision_info(all_messages)
        
        # Create batched processor inputs
        inputs = self.processor(
            text=all_text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        return inputs

    def forward(self, images_list, texts_list):
        """
        Forward pass to get batched token hidden states and attention masks.
        
        Args:
            images_list: List[List[PIL.Image]] - B lists of images (one per sample)
            texts_list: List[str] - B text prompts (one per sample)
        
        Returns:
            H: last_hidden_state [B, L_max, 2048]
            attention_mask: [B, L_max]
        """
        # Pack batched inputs
        inputs = self._pack_batch(images_list, texts_list)
        
        # Move to correct device (use backbone parameters since we don't have proj layer yet)
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get hidden states from backbone
        out = self.backbone(
            **inputs,
            output_hidden_states=True,  # Need this to get hidden states
            return_dict=True
        )
        
        # Extract last hidden state from the tuple of all layer outputs
        H = out.hidden_states[-1]  # [B, L, 2048]
        attention_mask = inputs.get("attention_mask", None)  # [B, L]
        return H, attention_mask