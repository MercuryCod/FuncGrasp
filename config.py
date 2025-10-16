"""
Configuration module for functional grasp training.
Contains all hyperparameters and settings.
"""

import torch
import os
import copy


class Config:
    """Configuration class with all training parameters."""
    
    # Contact class definitions
    # IMPORTANT: This order MUST match utils/contact_utils.py FINGER_IDS
    # Dataset returns: 0=no_contact, 1=palm, 2=thumb, 3=index, 4=middle, 5=ring, 6=pinky
    CONTACT_CLASSES = ["no_contact", "palm", "thumb", "index", "middle", "ring", "pinky"]
    NO_CONTACT_INDEX = 0  # Changed from 6 to match dataset
    NUM_CONTACT_CLASSES = 7
    
    # Model architecture
    MODEL = {
        'CSEM': 256,      # Semantic feature dimension
        'CGEO': 256,      # Geometric feature dimension
        # CFUSE = CSEM + CGEO = 512 (automatically computed in model)
        'DPOSE': 61,      # MANO pose dimension: 48 (pose) + 10 (shape) + 3 (trans)
        # MANO Parameter Breakdown:
        #   48: Axis-angle representation of 16 hand joints (excluding root)
        #   10: PCA shape coefficients
        #    3: Global wrist translation in meters [x, y, z]
        # Note: Do NOT confuse with joint positions (21 × 3 = 63)
        'K_CONTACT': 7,   # Contact classes (5 fingers + palm + no_contact)
        'n_points': 1024, # Points per object
        'qwen_tuning': os.environ.get('QWEN_TUNING', 'lora'),  # 'frozen', 'full', or 'lora'
        'contact_regression': True,  # Use class-based regression (BCE) instead of CE
        'inference_threshold': 0.4,  # Threshold for no_contact prediction in regression mode
    }
    
    # LoRA configuration (only used when qwen_tuning='lora')
    LORA = {
        'r': 32,                    # LoRA rank
        'alpha': 64,                # LoRA alpha (scaling factor)
        'dropout': 0.05,            # LoRA dropout
        'target_modules': [         # Which modules to apply LoRA to
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],
        'bias': 'none',             # 'none', 'all', or 'lora_only'
    }
    
    # Training parameters
    # Debug mode: enable frequent logging and checkpointing
    DEBUG_MODE = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    TRAINING = {
        'batch_size': 256,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'epochs': 100,
        'gradient_clip': 1.0,
        'lambda_contact': 1.5,  # Contact loss weight
        'lambda_flow': 1.0,     # Flow matching loss weight
        'log_interval': 1 if DEBUG_MODE else 20,     # Log every N batches
        'checkpoint_interval': 5 if DEBUG_MODE else 500,  # Save checkpoint every N batches
        'gradient_accumulation': 1,
    }
    
    # Evaluation settings
    EVAL = {
        'val_interval': 5 if DEBUG_MODE else 500,  # Validate every N training steps (batches)
        'num_qualitative': 3,  # Number of test samples to visualize in 3D
        'max_val_samples': 50 if DEBUG_MODE else None,  # Limit validation samples in debug mode (None = all)
    }
    
    # Data parameters
    DATA = {
        'root_dir': os.path.join(os.environ.get('DATA_ROOT', '/workspace/data'), 'OakInk'),
        'render_dir': os.path.join(os.environ.get('DATA_ROOT', '/workspace/data'), 'OakInk', 'rendered_objects'),
        'contact_threshold': 0.01,  # 10mm threshold for hard contact labels
        'soft_target_tau': 0.01,    # 10mm temperature for Gaussian kernel in soft targets
    }
    
    # CPU/GPU settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    
    # CPU-specific overrides (used when GPU not available)
    CPU_CONFIG = {
        'batch_size': 2,
        'num_workers': 0,
        'n_points': 256,
        'gradient_accumulation': 4
    }
    
    # Flow matching parameters
    FLOW = {
        'num_steps_train': 1,  # Single step during training (rectified flow)
        'num_steps_inference': 20,  # Integration steps for inference
        'ode_solver': 'euler',  # ODE solver type
    }
    
    # Experiment naming (can be overridden via ENV EXP_NAME)
    EXP_NAME = os.environ.get('EXP_NAME', 'default')

    # Paths (single source of truth, nested under ./exp/EXP_NAME)
    _BASE_EXP_DIR = os.path.join('./exp', EXP_NAME)
    PATHS = {
        'base_dir': _BASE_EXP_DIR,
        'checkpoint_dir': os.path.join(_BASE_EXP_DIR, 'checkpoints'),
        'log_dir': os.path.join(_BASE_EXP_DIR, 'logs'),
        'qual_dir': os.path.join(_BASE_EXP_DIR, 'visualizations'),  # Qualitative visualizations
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
            'flow': copy.deepcopy(cls.FLOW),
            'paths': copy.deepcopy(cls.PATHS),
            'eval': copy.deepcopy(cls.EVAL),
            'lora': copy.deepcopy(cls.LORA),
            'debug_mode': cls.DEBUG_MODE,
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
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)
