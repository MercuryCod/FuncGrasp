# Qwen3-VL Model Documentation

**Version**: 2.0  
**Date**: October 2025  
**Model**: Qwen3-VL-4B-Instruct

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

**Qwen3-VL-4B-Instruct** is a 4-billion-parameter vision-language model developed by Alibaba's Qwen Team. It represents the latest advancement in multimodal AI, capable of understanding both images and text to perform a wide range of vision-language tasks.

### Model Information

- **Model Name**: `Qwen/Qwen3-VL-4B-Instruct`
- **Parameters**: ~4 billion
- **Architecture**: Vision-Language Transformer
- **Hidden Size**: 2560
- **Release Date**: 2025
- **License**: Apache 2.0
- **Hugging Face**: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

### Why Qwen3-VL for Functional Grasp Prediction?

1. **Enhanced Visual Understanding**: Improved object recognition, layout analysis, and spatial reasoning
2. **Efficient**: 4B parameters provide excellent performance-to-cost ratio
3. **Instruction Following**: Advanced fine-tuning for natural language instruction comprehension
4. **Flexible Input**: Handles images of various resolutions and aspect ratios
5. **Integration**: Seamless integration with Hugging Face Transformers
6. **Numerical Stability**: Better handling of large feature spaces with bfloat16

---

## Key Features

### 1. Enhanced Visual Understanding

Qwen2.5-VL excels at:
- **Object Recognition**: Identifying common objects in various contexts
- **Text Recognition**: Reading and understanding text within images
- **Chart Analysis**: Interpreting charts, graphs, and data visualizations
- **Layout Understanding**: Analyzing document layouts and spatial arrangements
- **Fine-grained Details**: Recognizing small objects and subtle visual features

### 2. Agentic Capabilities

- **Visual Reasoning**: Multi-step reasoning based on visual information
- **Instruction Following**: Precise execution of complex visual tasks
- **Interactive Dialogue**: Maintaining context across multiple turns

### 3. Dynamic Resolution

- **Flexible Input**: Handles images at various resolutions (up to 4K)
- **Adaptive Processing**: Automatically adjusts to input image dimensions
- **Efficient Encoding**: Optimized vision encoder for fast processing

### 4. Video Understanding (Future Capability)

While our current implementation focuses on single images, Qwen2.5-VL supports:
- Long video comprehension
- Event capturing across frames
- Temporal reasoning

### 5. Structured Output Generation

- **Format Control**: Can generate outputs in specific formats (JSON, XML, etc.)
- **Visual Localization**: Can provide bounding boxes and spatial coordinates
- **Detailed Descriptions**: Rich, structured descriptions of visual content

---

## Model Architecture

### Overview

```
Input Images → Vision Encoder → Cross-Attention → Language Model → Hidden States
     ↓                                                                    ↓
Text Input  → Tokenizer ────────────────────────────────────────→ Output Features
```

### Components

#### 1. Vision Encoder
- **Type**: Efficient Vision Transformer (ViT)
- **Input**: RGB images (various resolutions)
- **Output**: Visual feature tokens
- **Key Features**:
  - Dynamic resolution support
  - Efficient patch embedding
  - Hierarchical feature extraction

#### 2. Language Model
- **Base**: Qwen2.5 (3B variant)
- **Architecture**: Transformer decoder
- **Hidden Size**: 2048
- **Layers**: 32
- **Attention Heads**: 32
- **Vocabulary**: 151,643 tokens

#### 3. Cross-Modal Fusion
- **Method**: Cross-attention between visual and text tokens
- **Position**: Early fusion in the decoder
- **Benefits**: Rich multimodal representations

### Training Strategy

1. **Pretraining**: Large-scale vision-language pairs
2. **Instruction Tuning**: Supervised fine-tuning on instruction-following datasets
3. **RLHF** (optional): Reinforcement learning from human feedback

---

## Installation

### Requirements

```bash
# Core dependencies
transformers >= 4.37.0
torch >= 2.0.0
torchvision >= 0.15.0
pillow >= 10.0.0
qwen-vl-utils >= 0.0.1

# Optional (for acceleration)
flash-attn >= 2.0.0  # For faster attention
accelerate >= 0.20.0  # For multi-GPU support
```

### Install from PyPI

```bash
# Basic installation
pip install transformers torch torchvision pillow

# Install qwen-vl-utils
pip install qwen-vl-utils

# Optional: Flash Attention (for 2x speed)
pip install flash-attn --no-build-isolation
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/QwenLM/Qwen2.5-VL.git
cd Qwen2.5-VL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

---

## Usage Guide

### Basic Usage

#### 1. Load Model and Processor

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically distribute across GPUs
)

# Load processor
processor = AutoProcessor.from_pretrained(model_name)
```

#### 2. Prepare Inputs

```python
from qwen_vl_utils import process_vision_info

# Load image
image = Image.open("path/to/image.jpg")

# Create message format
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image in detail."}
    ]
}]

# Process vision info
image_inputs, video_inputs = process_vision_info(messages)

# Apply chat template
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and prepare inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)
```

#### 3. Generate Response

```python
# Generate
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False  # Deterministic generation
)

# Decode
output_text = processor.batch_decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(output_text)
```

### Feature Extraction (Our Use Case)

For functional grasp prediction, we extract hidden states instead of generating text:

```python
import torch
import torch.nn as nn

class QwenFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        self.model.eval()
        
    def extract_features(self, images, texts):
        """
        Extract multimodal features.
        
        Args:
            images: List[PIL.Image] - batch of images
            texts: List[str] - batch of text prompts
        
        Returns:
            features: [B, L, hidden_size] - hidden states
            attention_mask: [B, L] - attention mask
        """
        # Prepare messages
        messages = []
        for img, text in zip(images, texts):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text}
                ]
            }])
        
        # Process
        image_inputs, video_inputs = process_vision_info(messages)
        text_inputs = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        
        inputs = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Forward pass (no generation)
        with torch.no_grad():
            outputs = self.model.model(  # Access the base model
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract last hidden state
        hidden_states = outputs.last_hidden_state  # [B, L, 2048]
        attention_mask = inputs.get("attention_mask", None)
        
        return hidden_states, attention_mask
```

### Multi-Image Input

```python
# Multiple images per sample
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
        {"type": "text", "text": "Compare these two images."}
    ]
}]
```

### Batch Processing

```python
# Process multiple samples at once
batch_messages = [
    [{"role": "user", "content": [{"type": "image", "image": img1}, {"type": "text", "text": prompt1}]}],
    [{"role": "user", "content": [{"type": "image", "image": img2}, {"type": "text", "text": prompt2}]}],
]

# Process batch
image_inputs, video_inputs = process_vision_info(batch_messages)
texts = [processor.apply_chat_template(msg, tokenize=False) for msg in batch_messages]
inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
```

---

## API Reference

### Model Classes

#### `Qwen2_5_VLForConditionalGeneration`

Main model class for vision-language tasks.

**Methods**:
- `forward(**inputs) -> ModelOutput`
  - Forward pass through the model
  - Returns: `ModelOutput` with `logits`, `hidden_states`, etc.
  
- `generate(**inputs, **generation_config) -> torch.Tensor`
  - Generate text based on vision-language inputs
  - Returns: Token IDs

**Attributes**:
- `config`: Model configuration
- `model`: Base Qwen2.5-VL model
- `lm_head`: Language modeling head

### Processor

#### `AutoProcessor`

Handles tokenization and image preprocessing.

**Methods**:
- `__call__(text, images, videos, padding, return_tensors)`
  - Process text and vision inputs
  - Returns: Dictionary of input tensors
  
- `apply_chat_template(messages, tokenize, add_generation_prompt)`
  - Format messages into chat template
  - Returns: Formatted string or token IDs

### Utilities

#### `qwen_vl_utils.process_vision_info(messages)`

Extract and process images/videos from message format.

**Args**:
- `messages`: List of message dictionaries

**Returns**:
- `image_inputs`: List of PIL Images
- `video_inputs`: List of video paths (or None)

---

## Performance Benchmarks

### Image Understanding Benchmarks

| Benchmark | Qwen2.5-VL-3B | Notes |
|-----------|---------------|-------|
| **General VQA** | | |
| VQAv2 | 78.2 | Visual Question Answering |
| TextVQA | 72.5 | Text-based VQA |
| DocVQA | 85.3 | Document Understanding |
| **Object Recognition** | | |
| COCO Detection | 45.2 mAP | Object detection capability |
| RefCOCO | 82.1 | Visual grounding |
| **Spatial Reasoning** | | |
| GQA | 62.8 | Compositional reasoning |
| NLVR2 | 75.4 | Visual reasoning |

### Computational Performance

| Metric | Value | Configuration |
|--------|-------|---------------|
| **Inference Speed** | | |
| Single Image (512×512) | ~150ms | H100, FP16, batch=1 |
| Batch Processing (BS=8) | ~900ms | H100, FP16 |
| **Memory Usage** | | |
| Model Weights | 6 GB | FP16 |
| Peak Inference (BS=1) | 8 GB | FP16, 512×512 image |
| Peak Inference (BS=8) | 24 GB | FP16, 512×512 images |

### Comparison with Other Models

| Model | Params | VQAv2 | Speed | Memory |
|-------|--------|-------|-------|--------|
| Qwen2.5-VL-3B | 3B | 78.2 | ⚡⚡⚡ | 8 GB |
| LLaVA-1.5-7B | 7B | 79.5 | ⚡⚡ | 16 GB |
| InstructBLIP-7B | 7B | 75.8 | ⚡⚡ | 18 GB |
| BLIP-2-2.7B | 2.7B | 65.4 | ⚡⚡⚡ | 6 GB |

**Verdict**: Qwen2.5-VL-3B offers the best performance-efficiency trade-off for our use case.

---

## Best Practices

### 1. Image Preprocessing

```python
from PIL import Image

# Recommended image loading
image = Image.open(path).convert('RGB')  # Always convert to RGB

# No manual resizing needed - model handles it
# The processor automatically resizes to optimal resolution
```

### 2. Memory Optimization

```python
# Enable gradient checkpointing (saves ~40% memory)
model.gradient_checkpointing_enable()

# Use FP16 for inference
model = model.to(dtype=torch.float16)

# Use flash attention if available (2x faster, less memory)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # Requires flash-attn package
)
```

### 3. Efficient Batching

```python
# Use DataLoader with custom collation
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Custom collation for Qwen2.5-VL"""
    images = [sample['image'] for sample in batch]
    texts = [sample['text'] for sample in batch]
    
    # Prepare messages
    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": text}
        ]}]
        for img, text in zip(images, texts)
    ]
    
    # Process batch
    image_inputs, video_inputs = process_vision_info(messages)
    text_inputs = [processor.apply_chat_template(msg, tokenize=False) for msg in messages]
    inputs = processor(text=text_inputs, images=image_inputs, videos=video_inputs, 
                      padding=True, return_tensors="pt")
    
    return inputs

dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
```

### 4. Fine-Tuning Tips

```python
# Use LoRA for efficient fine-tuning
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=64,  # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~50M (1.7% of total)
```

### 5. Prompt Engineering

**Good Prompts**:
```python
# Specific and descriptive
"Describe the spatial relationship between the hand and the mug."
"Identify which fingers are touching the object."
"Analyze the grasp type used to hold this tool."

# With constraints
"List the contact points as a JSON array with finger names."
"Provide a single-word answer: 'contact' or 'no-contact'."
```

**Bad Prompts**:
```python
# Too vague
"What do you see?"
"Describe the image."

# Too complex
"Analyze the hand pose, identify all contact points, classify the grasp type, 
 estimate the force distribution, and predict the intention."  # Split into multiple queries
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: `No module named 'qwen_vl_utils'`

**Solution**:
```bash
pip install qwen-vl-utils
```

#### 2. CUDA Out of Memory

**Solutions**:
```python
# Option 1: Reduce batch size
batch_size = 4  # Instead of 8

# Option 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Option 3: Use CPU offloading
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",  # Offload to disk
    offload_state_dict=True
)

# Option 4: Use smaller images
# Resize images to 256×256 or 384×384 before processing
```

#### 3. Slow Inference

**Solutions**:
```python
# Enable flash attention
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)

# Use FP16
model = model.to(dtype=torch.float16)

# Disable gradient computation
with torch.no_grad():
    outputs = model(**inputs)

# Use torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

#### 4. Model Not Loading Correctly

**Check**:
```python
# Verify model is downloaded
from transformers import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")

# Check device placement
print(model.device)  # Should show 'cuda:0' if GPU available

# Verify model architecture
print(model)
print(f"Hidden size: {model.config.hidden_size}")  # Should be 2048
```

#### 5. Hidden States Not Extracting Correctly

**Solution**:
```python
# Access the base model correctly
def get_encoder_module(model):
    """Navigate to the encoder module"""
    module = model
    
    # Unwrap PEFT if using LoRA
    if hasattr(module, 'base_model'):
        module = module.base_model
    
    # Get the core model
    if hasattr(module, 'model'):
        module = module.model  # Qwen2_5_VLModel
    
    return module

# Use it
encoder = get_encoder_module(model)
outputs = encoder(**inputs, output_hidden_states=True)
hidden_states = outputs.last_hidden_state
```

---

## Integration with Our Project

### Current Usage

In our Functional Grasp project, Qwen2.5-VL is used for semantic understanding:

```python
# File: models/semantics_qwen.py

class QwenSemanticsEncoder(nn.Module):
    """
    Semantic encoder using Qwen2.5-VL-3B-Instruct.
    Extracts multimodal features from images and text instructions.
    """
    def __init__(self, device=None, dtype=torch.float16, tuning='lora', lora_cfg=None):
        super().__init__()
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=dtype, 
            device_map=None
        )
        
        # Configure for feature extraction
        self.backbone.gradient_checkpointing_enable()
        self.hidden_size = self.backbone.config.hidden_size  # 2048
        
        # Setup tuning mode (lora or full)
        # ... (see models/semantics_qwen.py for full implementation)
    
    def forward(self, images_list, texts_list):
        """
        Extract semantic features.
        
        Args:
            images_list: List[List[PIL.Image]] - batch of image lists
            texts_list: List[str] - batch of instructions
        
        Returns:
            H: [B, L, 2048] - hidden states
            attention_mask: [B, L] - attention mask
        """
        # Pack batch using qwen_vl_utils
        inputs = self._pack_batch(images_list, texts_list)
        
        # Forward pass
        encoder_module = self._get_encoder_module()
        outputs = encoder_module(**inputs, output_hidden_states=True, return_dict=True)
        
        # Extract features
        H = outputs.last_hidden_state
        attention_mask = inputs.get("attention_mask", None)
        
        return H, attention_mask
```

### Pipeline Flow

```
Object Images (PIL) → Qwen2.5-VL → Semantic Features [B, L, 2048]
Text Instruction    ↗                            ↓
                                        Projection Layer
                                                 ↓
                                     Global Semantic Vector [B, 256]
                                                 ↓
                                          (Fused with PointNet++)
                                                 ↓
                                    Contact Prediction + Grasp Generation
```

---

## References

### Official Resources

- **Model Card**: [Qwen2.5-VL-3B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **GitHub Repository**: [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- **Documentation**: [Qwen Technical Documentation](https://qwen.readthedocs.io/)
- **Blog Post**: [Introducing Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/)

### Papers

- **Qwen2 Technical Report** (2024): Architecture and training details
- **Qwen-VL**: Original vision-language model paper

### Community

- **Discord**: [Qwen Community](https://discord.gg/qwen)
- **Discussions**: [Hugging Face Discussions](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions)

### Related Models

- **Qwen2.5-VL-7B-Instruct**: Larger variant (better performance, more memory)
- **Qwen2-VL-2B-Instruct**: Previous generation (smaller but less capable)
- **Qwen2.5-3B-Instruct**: Text-only variant

---

**Last Updated**: October 15, 2024  
**Model Version**: Qwen2.5-VL-3B-Instruct  
**Documentation Version**: 1.0

