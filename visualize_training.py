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
    val_macro_f1 = []
    val_per_class_acc = []
    
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
            
            # Parse validation header
            if 'Validation at step' in line:
                step_match = re.search(r'Validation at step (\d+):', line)
                if step_match:
                    current_val_step = int(step_match.group(1))
            
            # Parse validation metrics (new format)
            # Format: Loss: 1.6375 | Contact: 0.6298 | Flow: 1.0077
            if 'Loss:' in line and 'Contact:' in line and 'Flow:' in line:
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                contact_match = re.search(r'Contact: ([\d.]+)', line)
                flow_match = re.search(r'Flow: ([\d.]+)', line)
                
                if loss_match and contact_match and flow_match:
                    if 'current_val_step' in locals():
                        val_steps.append(current_val_step)
                        val_loss.append(float(loss_match.group(1)))
                        val_contact.append(float(contact_match.group(1)))
                        val_flow.append(float(flow_match.group(1)))
            
            # Parse overall accuracy and macro-F1
            # Format: Overall Acc: 0.8542 | Macro-F1: 0.3421
            if 'Overall Acc:' in line and 'Macro-F1:' in line:
                acc_match = re.search(r'Overall Acc: ([\d.]+)', line)
                f1_match = re.search(r'Macro-F1: ([\d.]+)', line)
                
                if acc_match and f1_match:
                    val_acc.append(float(acc_match.group(1)))
                    val_macro_f1.append(float(f1_match.group(1)))
            
            # Parse per-class accuracy
            # Format: Per-class Acc: ['0.123', '0.456', ...]
            if 'Per-class Acc:' in line:
                # Extract list of floats
                acc_list_match = re.search(r"Per-class Acc: \[(.*?)\]", line)
                if acc_list_match:
                    acc_str = acc_list_match.group(1)
                    # Parse individual values
                    accs = [float(x.strip().strip("'\"")) for x in acc_str.split(',')]
                    val_per_class_acc.append(accs)
            
            # Legacy format fallback
            # Format: Validation metrics: {'val_loss': 1.637544842866751, ...}
            if 'Validation metrics:' in line:
                # Extract the previous step number
                if train_steps and len(val_steps) < len(val_acc):
                    if len(val_steps) == 0 or val_steps[-1] != train_steps[-1]:
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
            'macro_f1': np.array(val_macro_f1) if val_macro_f1 else np.array([]),
            'per_class_acc': val_per_class_acc  # List of lists
        }
    }


def plot_metrics(data, output_path='outputs/training_metrics.png'):
    """Create training visualization plots."""
    # Determine if we have enhanced metrics
    has_enhanced = len(data['val'].get('macro_f1', [])) > 0
    
    if has_enhanced:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    else:
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
    
    # Add macro-F1 on secondary axis if available
    if has_enhanced and len(val['macro_f1']) > 0:
        ax2 = ax.twinx()
        ax2.plot(val['steps'], val['macro_f1'], 'g--', linewidth=2, marker='s', markersize=4, label='Val Macro-F1')
        ax2.set_ylabel('Macro-F1', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='lower right')
        ax2.set_ylim([0, 1])
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Contact Accuracy')
    ax.set_title('Contact Classification Accuracy')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 5 & 6: Enhanced metrics (if available)
    if has_enhanced:
        # Plot 5: Macro-F1 dedicated plot
        ax = axes[2, 0]
        if len(val['macro_f1']) > 0:
            ax.plot(val['steps'], val['macro_f1'], 'g-', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('Step')
            ax.set_ylabel('Macro-F1')
            ax.set_title('Validation Macro-F1 (Class-Balanced)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Plot 6: Per-class accuracy
        ax = axes[2, 1]
        if len(val['per_class_acc']) > 0:
            per_class_array = np.array(val['per_class_acc'])  # [num_validations, 7]
            class_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little', 'Palm', 'No Contact']
            colors = ['#FF0000', '#FF8C00', '#FFD700', '#00FF00', '#0000FF', '#9400D3', '#C0C0C0']
            
            for class_id in range(min(7, per_class_array.shape[1])):
                ax.plot(val['steps'][:len(per_class_array)], 
                       per_class_array[:, class_id], 
                       linewidth=2, marker='o', markersize=3,
                       label=class_names[class_id],
                       color=colors[class_id],
                       alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Per-Class Accuracy')
            ax.set_title('Validation Per-Class Accuracy')
            ax.legend(loc='best', fontsize=8, ncol=2)
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
    
    if has_enhanced and len(val['macro_f1']) > 0:
        print(f"  Macro-F1: {val['macro_f1'][-1]:.3f} (started: {val['macro_f1'][0]:.3f})")
    
    print(f"\nBest val metrics:")
    best_idx = np.argmin(val['loss'])
    print(f"  Best val loss: {val['loss'][best_idx]:.4f} at step {val['steps'][best_idx]}")
    best_idx = np.argmin(val['flow'])
    print(f"  Best val flow: {val['flow'][best_idx]:.4f} at step {val['steps'][best_idx]}")
    best_idx = np.argmax(val['acc'])
    print(f"  Best val acc: {val['acc'][best_idx]:.3f} at step {val['steps'][best_idx]}")
    
    if has_enhanced and len(val['macro_f1']) > 0:
        best_idx = np.argmax(val['macro_f1'])
        print(f"  Best macro-F1: {val['macro_f1'][best_idx]:.3f} at step {val['steps'][best_idx]}")
    
    plt.close()


if __name__ == "__main__":
    log_path = "logs/run.log"
    data = parse_log(log_path)
    plot_metrics(data)

