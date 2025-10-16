"""
Training utilities: logging, checkpointing, metrics, visualization.

Provides essential functions for:
- Setting up logging to both file and console
- Saving and loading model checkpoints
- Computing evaluation metrics for contact prediction
- Visualizing model predictions during validation
"""
import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from utils.visualize import visualize_grasp
from utils.contact_utils import compute_contact_points


def setup_logging(log_dir: str, exp_name: str = "train"):
    """
    Setup logging to both file and console with timestamps.
    
    Args:
        log_dir: Directory to save log files
        exp_name: Experiment name for log file prefix
    
    Returns:
        Path to created log file
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{exp_name}_{timestamp}.log')
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )
    
    logging.info(f"Logging initialized. Output file: {log_file}")
    
    # Create a symlink to the latest log for easy access
    latest_log = os.path.join(log_dir, 'run.log')
    if os.path.islink(latest_log) or os.path.exists(latest_log):
        os.remove(latest_log)
    os.symlink(os.path.basename(log_file), latest_log)
    
    return log_file


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    is_best: bool = False
):
    """
    Save training checkpoint with model, optimizer, and scheduler states.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch number
        metrics: Dict of metrics (for logging purposes)
        path: Save path for checkpoint
        is_best: If True, also save a copy as 'best.pth'
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved: {path}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Best checkpoint saved: {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load training checkpoint and restore model/optimizer/scheduler states.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map tensors to
    
    Returns:
        Checkpoint dictionary containing epoch, metrics, etc.
    """
    logging.info(f"Loading checkpoint from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    logging.info(f"Checkpoint loaded successfully (epoch {epoch})")
    
    return checkpoint


def compute_contact_metrics(
    logits: torch.Tensor, 
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for contact prediction, including per-class accuracy.
    
    Args:
        logits: [B, N, 7] predicted logits from contact head
            Index 0: no_contact
            Index 1-6: palm, thumb, index, middle, ring, pinky
        labels: [B, N] ground truth labels (0-6)
        threshold: Sigmoid threshold for binary contact prediction (default 0.5)
    
    Returns:
        Dict with:
            - contact_accuracy: Overall accuracy for contact vs non-contact
            - contact_precision: Precision for contact prediction
            - contact_recall: Recall for contact prediction
            - contact_f1: F1 score for contact prediction
            - acc_no_contact, acc_palm, acc_thumb, acc_index, acc_middle, acc_ring, acc_pinky: Per-class accuracy
    """
    with torch.no_grad():
        # Binary contact prediction
        part_probs = torch.sigmoid(logits[..., 1:])  # [B, N, 6]
        pred_contact = (part_probs > threshold).any(dim=-1)  # [B, N]
        true_contact = labels > 0  # [B, N]
        
        # Compute confusion matrix elements
        tp = (pred_contact & true_contact).sum().float()
        fp = (pred_contact & ~true_contact).sum().float()
        tn = (~pred_contact & ~true_contact).sum().float()
        fn = (~pred_contact & true_contact).sum().float()
        
        # Compute metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics = {
            'contact_accuracy': accuracy.item(),
            'contact_precision': precision.item(),
            'contact_recall': recall.item(),
            'contact_f1': f1.item()
        }
        
        # Per-class accuracy aligned with multi-label modeling
        # Derive predicted class as: no_contact if no part exceeds threshold; else argmax over parts
        K = logits.shape[-1]
        part_probs = torch.sigmoid(logits[..., 1:])  # [B, N, 6]
        has_contact = (part_probs > threshold).any(dim=-1)  # [B, N]
        # Argmax over parts (1..6), then map to 1..6; if no contact, set 0
        part_argmax = part_probs.argmax(dim=-1) + 1  # in 1..6
        preds_mll = torch.where(has_contact, part_argmax, torch.zeros_like(part_argmax))  # 0..6
        
        class_names = ['no_contact', 'palm', 'thumb', 'index', 'middle', 'ring', 'pinky']
        for class_id, class_name in enumerate(class_names):
            class_mask = (labels == class_id)
            if class_mask.sum() > 0:
                correct = (preds_mll[class_mask] == class_id).sum().float()
                total = class_mask.sum().float()
                class_acc = (correct / total).item()
            else:
                class_acc = 0.0
            metrics[f'acc_{class_name}'] = class_acc
        
        return metrics


def visualize_predictions(
    batch: Dict,
    outputs: Dict,
    save_dir: str,
    num_samples: int = 3
):
    """
    Visualize model predictions in 3D for qualitative evaluation.
    
    Creates side-by-side visualizations showing:
    - Object point cloud colored by predicted contact probabilities
    - Predicted hand mesh with predicted pose
    - Ground truth hand mesh for comparison
    
    Args:
        batch: Input batch from dataloader
        outputs: Model outputs including 'logits_c' and 'predicted_poses'
        save_dir: Directory to save visualization HTML files
        num_samples: Number of samples to visualize from batch (default: 3)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data
    pts = batch['pts'].cpu()  # [B, N, 3]
    mano_params = batch['mano_params'].cpu()  # [B, 61]
    contact_labels_gt = batch['contact_hard_labels'].cpu()  # [B, N] - GT hard labels
    logits_c = outputs['logits_c'].cpu()  # [B, N, 7] - predictions
    predicted_poses = outputs.get('predicted_poses', mano_params).cpu()  # [B, 61] - predicted MANO params
    
    B = min(num_samples, pts.shape[0])
    
    for i in range(B):
        # Extract single sample
        obj_points = pts[i].numpy()  # [N, 3]
        
        # Ground truth MANO parameters
        pose_gt = mano_params[i, :48].numpy()
        shape_gt = mano_params[i, 48:58].numpy()
        trans_gt = mano_params[i, 58:61].numpy()
        
        # Predicted MANO parameters from flow matching
        pose_pred = predicted_poses[i, :48].numpy()
        shape_pred = predicted_poses[i, 48:58].numpy()
        trans_pred = predicted_poses[i, 58:61].numpy()
        
        # Predicted contact (from logits)
        probs_all = torch.softmax(logits_c[i], dim=-1)  # [N, 7]
        pred_labels = probs_all.argmax(dim=-1).numpy()  # [N] - predicted class
        
        # Create contact info dict for predicted contacts
        contact_info_pred = {
            'finger_labels': pred_labels,  # Use predicted labels from model
            'contact_mask': pred_labels > 0,
        }
        
        # Create contact info dict for GT contacts
        contact_info_gt = {
            'finger_labels': contact_labels_gt[i].numpy(),  # Use GT labels from dataset
            'contact_mask': contact_labels_gt[i].numpy() > 0,
        }
        
        # Get metadata for naming
        category = batch['metadata']['categories'][i]
        intent = batch['metadata']['intents'][i]
        shape_id = batch['metadata']['shape_ids'][i]
        
        # Save predicted visualization with PREDICTED pose and contacts
        pred_path = os.path.join(save_dir, f'sample_{i}_{category}_{shape_id}_predicted.html')
        visualize_grasp(
            pose_pred, shape_pred, trans_pred, obj_points,  # ← Now using predicted pose!
            output_path=pred_path,
            contact_info=contact_info_pred,
            show_joints=False
        )
        
        # Save GT visualization for comparison
        gt_path = os.path.join(save_dir, f'sample_{i}_{category}_{shape_id}_groundtruth.html')
        visualize_grasp(
            pose_gt, shape_gt, trans_gt, obj_points,
            output_path=gt_path,
            contact_info=contact_info_gt,
            show_joints=False
        )
    
    logging.info(f"All visualizations saved to: {save_dir}")

