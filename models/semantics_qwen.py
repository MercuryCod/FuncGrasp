"""
Qwen2.5-VL-3B-Instruct Semantics Encoder
Wraps Qwen2.5-VL-3B-Instruct and returns batched token hidden states H ∈ ℝ^{B×L_max×2048}
for downstream masked pooling. Supports frozen/full/lora tuning modes.
Each sample in the batch has its own text and set of images.
"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class QwenSemanticsEncoder(nn.Module):
    """
    Wraps Qwen2.5-VL-3B-Instruct to produce batched token hidden states H ∈ ℝ^{B×L_max×2048}.
    Supports three tuning modes:
    - 'frozen': Freeze entire backbone (default)
    - 'full': Train entire backbone end-to-end
    - 'lora': Apply LoRA adapters to specified modules
    
    Fixed model: Qwen/Qwen2.5-VL-3B-Instruct
    Output: last_hidden_state [B, L_max, 2048] where B is batch size
    Each sample in the batch has its own text prompt and set of images.
    """
    def __init__(
        self,
        min_pixels=None,
        max_pixels=None,
        device=None,
        dtype=torch.bfloat16,
        tuning='frozen',
        lora_cfg=None
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
        
        # Load 8-bit model if requested (requires bitsandbytes)
        load_in_8bit = False
        if lora_cfg and lora_cfg.get('use_8bit', False):
            try:
                import bitsandbytes  # noqa: F401
                load_in_8bit = True
            except ImportError:
                print("Warning: bitsandbytes not installed. Loading model in full precision.")
        
        # Use the *backbone* (no LM head) to access hidden states
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,  # Don't use device_map to avoid multi-GPU issues
            load_in_8bit=load_in_8bit
        )
        
        # Move to device if specified and not using 8-bit
        if device is not None and not load_in_8bit:
            self.backbone = self.backbone.to(device)
        
        # Apply tuning mode
        self.tuning = tuning
        if tuning == 'frozen':
            # Freeze all parameters
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("Qwen backbone frozen.")
        
        elif tuning == 'lora':
            # Freeze base model
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            
            # Apply LoRA
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare for 8-bit training if needed
            if load_in_8bit:
                self.backbone = prepare_model_for_kbit_training(self.backbone)
            
            # Configure LoRA
            lora_cfg = lora_cfg or {}
            lora_config = LoraConfig(
                r=lora_cfg.get('r', 16),
                lora_alpha=lora_cfg.get('alpha', 32),
                lora_dropout=lora_cfg.get('dropout', 0.05),
                target_modules=lora_cfg.get('target_modules', [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ]),
                bias=lora_cfg.get('bias', 'none'),
                task_type='CAUSAL_LM'
            )
            
            # Apply LoRA to model
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
            print(f"LoRA applied with r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        elif tuning == 'full':
            # Keep all parameters trainable
            print("Qwen backbone fully trainable.")
        
        else:
            raise ValueError(f"Unknown tuning mode: {tuning}. Choose from 'frozen', 'full', 'lora'")
        
        # Enable gradient checkpointing for memory efficiency (works with all modes)
        self.backbone.gradient_checkpointing_enable()
        
        # Store hidden size for reference
        # Qwen2.5-VL-3B-Instruct has hidden_size = 2048
        self.hidden_size = self.backbone.config.hidden_size  # 2048
        
        # Store model name for checkpoint saving
        self.model_name = model_name

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
    
    def save_lora_weights(self, path):
        """Save LoRA adapter weights if using LoRA mode."""
        if self.tuning == 'lora':
            self.backbone.save_pretrained(path)
            print(f"LoRA weights saved to {path}")
        else:
            print(f"Not in LoRA mode (current: {self.tuning}), skipping adapter save.")
    
    def load_lora_weights(self, path):
        """Load LoRA adapter weights if using LoRA mode."""
        if self.tuning == 'lora':
            # Note: This assumes backbone is already a PEFT model
            # In practice, you might need to reload base model first
            print(f"Loading LoRA weights from {path}")
            # from peft import PeftModel
            # self.backbone = PeftModel.from_pretrained(base_model, path)
        else:
            print(f"Not in LoRA mode (current: {self.tuning}), skipping adapter load.")