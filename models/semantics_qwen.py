# semantics_qwen.py
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenSemanticsEncoder(nn.Module):
    def __init__(self, tuning='lora', lora_cfg=None):
        super().__init__()
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )


        self.tuning = tuning

        if tuning == 'lora':
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            from peft import LoraConfig, get_peft_model
            lora_cfg = lora_cfg or {}
            # Defaults match Config.LORA for consistency
            lora_config = LoraConfig(
                r=lora_cfg.get('r', 32),
                lora_alpha=lora_cfg.get('alpha', 64),
                lora_dropout=lora_cfg.get('dropout', 0.05),
                target_modules=lora_cfg.get('target_modules', [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ]),
                bias=lora_cfg.get('bias', 'none'),
                task_type='CAUSAL_LM'
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        elif tuning == 'full':
            # Full fine-tuning: all parameters trainable
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown tuning mode: {tuning}. Only 'lora' or 'full' are supported.")

        # Enable gradient checkpointing for both tuning modes
        self.backbone.gradient_checkpointing_enable()
            
        self.hidden_size = self.backbone.config.hidden_size  # 2048 for Qwen2.5-VL-3B
        self.model_name = model_name

    def _pack_batch(self, images_list, texts_list):
        all_messages, all_text_inputs = [], []
        for images, text in zip(images_list, texts_list):
            content = [{"type": "image", "image": img} for img in (images or [])]
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            all_messages.append(messages)
            all_text_inputs.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
        image_inputs, video_inputs = process_vision_info(all_messages)
        inputs = self.processor(
            text=all_text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        return inputs

    def _get_encoder_module(self):
        module = self.backbone
        if hasattr(module, 'base_model'):
            module = module.base_model
        if hasattr(module, 'model'):
            module = module.model  # Qwen2_5_VLModel
        if hasattr(module, 'model'):
            module = module.model
        return module

    def forward(self, images_list, texts_list):
        inputs = self._pack_batch(images_list, texts_list)
        device = next(self.backbone.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        inputs['use_cache'] = False

        encoder_module = self._get_encoder_module()
        out = encoder_module(
            **inputs,
            output_hidden_states=True,   # ensure we get hidden states
            return_dict=True
        )

        if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
            H = out.last_hidden_state
        elif hasattr(out, 'hidden_states') and out.hidden_states is not None:
            H = out.hidden_states[-1]
        else:
            raise AttributeError("Qwen model output does not contain last_hidden_state or hidden_states")

        attention_mask = inputs.get("attention_mask", None)
        return H, attention_mask

    def save_lora_weights(self, path):
        if self.tuning == 'lora':
            self.backbone.save_pretrained(path)

    def load_lora_weights(self, path):
        """Load LoRA adapter weights from checkpoint."""
        if self.tuning == 'lora':
            from peft import PeftModel
            self.backbone = PeftModel.from_pretrained(self.backbone, path)
            print(f"Loaded LoRA weights from {path}")
        else:
            print(f"Warning: load_lora_weights called but tuning mode is '{self.tuning}'")
