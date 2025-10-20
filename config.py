"""
Configuration module for functional grasp training.
Contains all hyperparameters and settings.
"""

import os
import copy

import torch


class Config:
    """Configuration class with all training parameters."""

    # Contact class definitions
    # IMPORTANT: This order MUST match utils/contact_utils.py FINGER_IDS
    # Indexing convention (consistent throughout codebase):
    #   Index 0: no_contact (no hand part touching object point)
    #   Index 1-6: palm, thumb, index, middle, ring, pinky (6 contact parts)
    #
    # Dataset returns: 0=no_contact, 1=palm, 2=thumb, 3=index, 4=middle, 5=ring, 6=pinky
    # Model logits: [0, 1, 2, 3, 4, 5, 6] where index 0 is unused in loss
    # Soft targets: [0-5] mapping to logits[1:6] (no_contact excluded)
    CONTACT_CLASSES = ["no_contact", "palm", "thumb", "index", "middle", "ring", "pinky"]
    NO_CONTACT_INDEX = 0  # Class 0 = no_contact (canonical definition)
    NUM_CONTACT_CLASSES = 7

    # Model architecture
    MODEL = {
        'CSEM': 256,      # Semantic feature dimension
        'CGEO': 256,      # Geometric feature dimension
        # CFUSE = CSEM + CGEO = 512 (automatically computed in model)
        'DPOSE': 61,      # MANO parameter dimension (INPUT to MANO layer)
        # MANO Parameter Breakdown (61-dimensional INPUT vector):
        #   48: Axis-angle representation of 16 hand joints (excluding root)
        #   10: PCA shape coefficients (hand shape identity)
        #    3: Global wrist translation in meters [x, y, z]
        #
        # Note: Do NOT confuse DPOSE with MANO layer OUTPUT:
        #   - DPOSE = 61 (parameters that control the hand model)
        #   - Joint positions = 21 joints × 3 coords = 63 (OUTPUT of MANO forward kinematics)
        #   - Hand vertices = 778 vertices × 3 coords = 2334 (OUTPUT mesh geometry)
        'K_CONTACT': 7,   # Contact classes (5 fingers + palm + no_contact)
        'n_points': 1024, # Points per object
        'qwen_tuning': os.environ.get('QWEN_TUNING', 'lora'),  # 'lora' or 'full'
        'contact_regression': True,  # Use class-based regression (BCE) instead of CE
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
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'epochs': 100,
        'warmup_epochs': 3,  # LR warmup for first 3 epochs
        'gradient_clip': 1.0,
        'lambda_contact': 1.0,  # Contact loss weight
        'lambda_flow': 1.0,     # Flow matching loss weight
        'log_interval': 1 if DEBUG_MODE else 20,     # Log every N batches
        'checkpoint_interval': 5 if DEBUG_MODE else 500,  # Save checkpoint every N epochs
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
        
        # ═══════════════════════════════════════════════════════════════════
        # Contact Thresholds (Three Different Parameters, Different Purposes)
        # ═══════════════════════════════════════════════════════════════════
        
        # 1. GROUND TRUTH: Binary contact decision (distance-based)
        'contact_threshold': 0.01,  # 10mm - Used in dataset loading
                                     # Purpose: Create hard labels from hand-object distances
                                     # Logic: distance < 0.01m → in contact (label=1-6)
                                     #        distance >= 0.01m → no contact (label=0)
                                     # Used in: utils/contact_utils.py::compute_contact_points()
        
        # 2. SOFT TARGETS: RBF kernel scale (distance → probability)
        'soft_target_tau': 0.01,     # 10mm - Used in batch collation
                                     # Purpose: Convert distances to independent probabilities
                                     # Formula: prob = exp(-distance/tau) [NO normalization!]
                                     # Example: distance=10mm → prob=exp(-1)=0.37 (37%)
                                     # Used in: dataset/collate.py::compute_soft_targets_from_per_finger_distances()
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL DESIGN DECISION: Aligned Thresholds
        # ═══════════════════════════════════════════════════════════════════
        # contact_threshold and soft_target_tau are INTENTIONALLY set to the same value (0.01)
        # This ensures consistency between hard labels and soft probability distributions:
        #
        # At the threshold distance (10mm):
        #   - Hard label: Transitions from contact (1-6) to no_contact (0)
        #   - Soft probability: exp(-0.01/0.01) = exp(-1) ≈ 0.37 (37%)
        #
        # This alignment ensures smooth transitions:
        #   - Points below 10mm (contact): probability > 37%
        #   - Points above 10mm (no contact): probability < 37%
        #   - The model learns consistent boundaries between hard and soft supervision
        
        'no_contact_weight': 0.1,    # Weight for non-contact points in loss (handles class imbalance)
        'filter_zero_contact': True, # Filter out approach poses with no contact
    }

    # CPU/GPU settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0

    # CPU-specific overrides (used when GPU not available)
    CPU_CONFIG = {
        'batch_size': 2,
        'num_workers': 0,
        'n_points': 256,
    }

    # Flow matching parameters
    FLOW = {
        'num_steps_inference': 20,  # Number of Euler integration steps during sampling
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

        return config

    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)

