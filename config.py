"""
Configuration module for functional grasp training.
Contains all hyperparameters and settings.
"""

import torch
import os


class Config:
    """Configuration class with all training parameters."""
    
    # Model architecture
    MODEL = {
        'qwen_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'freeze_qwen': True,  # Set False to fine-tune Qwen backbone
        'CSEM': 256,      # Semantic feature dimension
        'CGEO': 256,      # Geometric feature dimension
        # CFUSE = CSEM + CGEO = 512 (automatically computed in model)
        'DPOSE': 28,      # Pose dimension (3 wrist + 25 joints)
        'K_CONTACT': 1,   # Contact classes (binary)
        'n_points': 1024, # Points per object
    }
    
    # Training parameters
    TRAINING = {
        'batch_size': 4 if torch.cuda.is_available() else 2,
        'learning_rate': 1e-4,  # For new layers
        'backbone_lr': 1e-5,    # For Qwen backbone (if unfrozen)
        'weight_decay': 0.05,
        'epochs': 100,
        'gradient_clip': 1.0,
        'lambda_contact': 1.0,  # Contact loss weight
        'lambda_flow': 1.0,     # Flow matching loss weight
        'log_interval': 10,     # Log every N batches
        'checkpoint_interval': 500,  # Save checkpoint every N batches
        'gradient_accumulation': 1,  # Increase if unfreezing Qwen
    }
    
    # Data parameters
    DATA = {
        'root_dir': os.environ.get('OAKINK_PATH', '/Users/mercury/Developer/FuncGrasp/OakInk'),
        'split_mode': 'split0',  # Object-based split
        'contact_threshold': 0.01,  # 1cm for contact approximation
        'use_cache': True,
        'single_view': True,  # Use single view for efficiency
        'view_idx': 0,  # Which view to use (0-3)
    }
    
    # CPU/GPU settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    FP16 = torch.cuda.is_available()  # Mixed precision only on GPU
    
    # CPU-specific settings
    CPU_CONFIG = {
        'batch_size': 2,
        'num_workers': 0,
        'n_points': 512,  # Reduce for CPU
        'gradient_accumulation': 4,  # Simulate larger batches
        'checkpoint_freq': 100,
    }
    
    # Fine-tuning specific settings (when freeze_qwen=False)
    FINETUNE_CONFIG = {
        'batch_size': 1,  # Very small batch for 3B model
        'gradient_accumulation': 8,  # Effective batch = 8
        'backbone_lr': 1e-5,  # Lower LR for pretrained weights
        'head_lr': 1e-4,  # Higher LR for new layers
        'use_fp16': True,  # Mixed precision essential
        'gradient_checkpointing': True,  # Save memory
    }
    
    # Flow matching parameters
    FLOW = {
        'num_steps_train': 1,  # Single step during training (rectified flow)
        'num_steps_inference': 20,  # Integration steps for inference
        'ode_solver': 'euler',  # ODE solver type
    }
    
    # Paths
    PATHS = {
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'output_dir': './outputs',
    }
    
    # Qwen processor settings
    QWEN = {
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1024 * 28 * 28,
        'device_map': 'auto',
        'dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    
    # Evaluation settings
    EVAL = {
        'val_interval': 100,  # Validate every N batches
        'num_val_samples': 50,  # Number of validation samples
        'num_poses_per_object': 4,  # M poses to generate per object
    }
    
    @classmethod
    def get_config(cls, mode='train'):
        """
        Get configuration dictionary.
        
        Args:
            mode: 'train' or 'eval'
        
        Returns:
            config dict
        """
        config = {
            'model': cls.MODEL,
            'training': cls.TRAINING,
            'data': cls.DATA,
            'device': cls.DEVICE,
            'num_workers': cls.NUM_WORKERS,
            'fp16': cls.FP16,
            'flow': cls.FLOW,
            'paths': cls.PATHS,
            'qwen': cls.QWEN,
            'eval': cls.EVAL,
        }
        
        # Override with CPU config if not using GPU
        if not torch.cuda.is_available():
            config['training']['batch_size'] = cls.CPU_CONFIG['batch_size']
            config['num_workers'] = cls.CPU_CONFIG['num_workers']
            config['model']['n_points'] = cls.CPU_CONFIG['n_points']
            config['training']['gradient_accumulation'] = cls.CPU_CONFIG['gradient_accumulation']
        
        return config
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        import os
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)