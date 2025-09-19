"""
Configuration module for functional grasp training.
Contains all hyperparameters and settings.
"""

import torch
import os
import copy


class Config:
    """Configuration class with all training parameters."""
    
    # Model architecture
    MODEL = {
        'CSEM': 256,      # Semantic feature dimension
        'CGEO': 256,      # Geometric feature dimension
        # CFUSE = CSEM + CGEO = 512 (automatically computed in model)
        'DPOSE': 63,      # Pose dimension (21 joints × 3 coordinates)
        'K_CONTACT': 1,   # Contact classes (binary)
        'n_points': 1024, # Points per object
    }
    
    # Training parameters
    TRAINING = {
        'batch_size': 4 if torch.cuda.is_available() else 2,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'epochs': 100,
        'gradient_clip': 1.0,
        'lambda_contact': 1.0,  # Contact loss weight
        'lambda_flow': 1.0,     # Flow matching loss weight
        'log_interval': 10,     # Log every N batches
        'checkpoint_interval': 500,  # Save checkpoint every N batches
        'gradient_accumulation': 1,
    }
    
    # Data parameters
    DATA = {
        'root_dir': os.environ.get('OAKINK_PATH', '/DATA/disk0/OakInk'),
        'render_dir': os.environ.get('OAKINK_RENDER_DIR', '/DATA/disk0/OakInk/rendered_objects'),
        'split_mode': 'split0',  # Object-based split
        'contact_threshold': 0.01,  # 1cm for contact approximation
        'use_cache': True,
        'single_view': True,  # Use single view for efficiency
    }
    
    # CPU/GPU settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    FP16 = torch.cuda.is_available()  # Mixed precision only on GPU
    
    # CPU-specific settings
    CPU_CONFIG = {
        'batch_size': 1,
        'num_workers': 0,
        'n_points': 512,  # Reduce for CPU
        'gradient_accumulation': 4,  # Simulate larger batches
        'checkpoint_freq': 100,
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
            'model': copy.deepcopy(cls.MODEL),
            'training': copy.deepcopy(cls.TRAINING),
            'data': copy.deepcopy(cls.DATA),
            'device': cls.DEVICE,
            'num_workers': cls.NUM_WORKERS,
            'fp16': cls.FP16,
            'flow': copy.deepcopy(cls.FLOW),
            'paths': copy.deepcopy(cls.PATHS),
            'eval': copy.deepcopy(cls.EVAL),
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
