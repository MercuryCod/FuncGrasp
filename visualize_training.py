"""
Visualize training metrics from run.log
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log(log_path):
    """Parse training log file and extract metrics."""
    train_steps = []
    train_loss = []
    train_contact = []
    train_flow = []
    train_acc = []
    
    val_steps = []
    val_loss = []
    val_contact = []
    val_flow = []
    val_acc = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse training steps
            # Format: Step 000000 | loss=3.0220 | contact=1.9930 | flow=1.0290 | acc=0.061
            match = re.search(r'Step (\d+) \| loss=([\d.]+) \| contact=([\d.]+) \| flow=([\d.]+) \| acc=([\d.]+)', line)
            if match:
                train_steps.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                train_contact.append(float(match.group(3)))
                train_flow.append(float(match.group(4)))
                train_acc.append(float(match.group(5)))
            
            # Parse validation metrics
            # Format: Validation metrics: {'val_loss': 1.637544842866751, 'val_contact_loss': 0.6297742449320279, ...}
            if 'Validation metrics:' in line:
                # Extract the previous step number
                if train_steps:
                    val_steps.append(train_steps[-1])
                
                # Parse the dict
                val_loss_match = re.search(r"'val_loss': ([\d.]+)", line)
                val_contact_match = re.search(r"'val_contact_loss': ([\d.]+)", line)
                val_flow_match = re.search(r"'val_flow_loss': ([\d.]+)", line)
                val_acc_match = re.search(r"'val_contact_acc': ([\d.]+)", line)
                
                if val_loss_match:
                    val_loss.append(float(val_loss_match.group(1)))
                if val_contact_match:
                    val_contact.append(float(val_contact_match.group(1)))
                if val_flow_match:
                    val_flow.append(float(val_flow_match.group(1)))
                if val_acc_match:
                    val_acc.append(float(val_acc_match.group(1)))
    
    return {
        'train': {
            'steps': np.array(train_steps),
            'loss': np.array(train_loss),
            'contact': np.array(train_contact),
            'flow': np.array(train_flow),
            'acc': np.array(train_acc),
        },
        'val': {
            'steps': np.array(val_steps),
            'loss': np.array(val_loss),
            'contact': np.array(val_contact),
            'flow': np.array(val_flow),
            'acc': np.array(val_acc),
        }
    }


def plot_metrics(data, output_path='outputs/training_metrics.png'):
    """Create training visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FuncGrasp Training Progress', fontsize=16, fontweight='bold')
    
    train = data['train']
    val = data['val']
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(train['steps'], train['loss'], 'b-', alpha=0.6, linewidth=1, label='Train')
    ax.plot(val['steps'], val['loss'], 'r-', linewidth=2, marker='o', markersize=4, label='Val')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Contact + Flow)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Contact Loss
    ax = axes[0, 1]
    ax.plot(train['steps'], train['contact'], 'b-', alpha=0.6, linewidth=1, label='Train')
    ax.plot(val['steps'], val['contact'], 'r-', linewidth=2, marker='o', markersize=4, label='Val')
    ax.set_xlabel('Step')
    ax.set_ylabel('Contact Loss')
    ax.set_title('Contact Loss (7-way Classification)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Flow Loss
    ax = axes[1, 0]
    ax.plot(train['steps'], train['flow'], 'b-', alpha=0.6, linewidth=1, label='Train')
    ax.plot(val['steps'], val['flow'], 'r-', linewidth=2, marker='o', markersize=4, label='Val')
    ax.set_xlabel('Step')
    ax.set_ylabel('Flow Loss')
    ax.set_title('Flow Matching Loss (63D Pose)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Contact Accuracy
    ax = axes[1, 1]
    ax.plot(train['steps'], train['acc'], 'b-', alpha=0.6, linewidth=1, label='Train')
    ax.plot(val['steps'], val['acc'], 'r-', linewidth=2, marker='o', markersize=4, label='Val')
    ax.set_xlabel('Step')
    ax.set_ylabel('Contact Accuracy')
    ax.set_title('Contact Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training visualization to {output_path}")
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Total steps: {train['steps'][-1]}")
    print(f"\nTrain metrics (final):")
    print(f"  Loss: {train['loss'][-1]:.4f} (started: {train['loss'][0]:.4f})")
    print(f"  Contact: {train['contact'][-1]:.4f} (started: {train['contact'][0]:.4f})")
    print(f"  Flow: {train['flow'][-1]:.4f} (started: {train['flow'][0]:.4f})")
    print(f"  Accuracy: {train['acc'][-1]:.3f} (started: {train['acc'][0]:.3f})")
    
    print(f"\nVal metrics (final):")
    print(f"  Loss: {val['loss'][-1]:.4f} (started: {val['loss'][0]:.4f})")
    print(f"  Contact: {val['contact'][-1]:.4f} (started: {val['contact'][0]:.4f})")
    print(f"  Flow: {val['flow'][-1]:.4f} (started: {val['flow'][0]:.4f})")
    print(f"  Accuracy: {val['acc'][-1]:.3f} (started: {val['acc'][0]:.3f})")
    
    print(f"\nBest val metrics:")
    best_idx = np.argmin(val['loss'])
    print(f"  Best val loss: {val['loss'][best_idx]:.4f} at step {val['steps'][best_idx]}")
    best_idx = np.argmin(val['flow'])
    print(f"  Best val flow: {val['flow'][best_idx]:.4f} at step {val['steps'][best_idx]}")
    best_idx = np.argmax(val['acc'])
    print(f"  Best val acc: {val['acc'][best_idx]:.3f} at step {val['steps'][best_idx]}")
    
    plt.close()


if __name__ == "__main__":
    log_path = "logs/run.log"
    data = parse_log(log_path)
    plot_metrics(data)

