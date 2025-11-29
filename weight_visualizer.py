"""
Weight Visualizer for Neural Network
Real-time heatmap visualization of weight matrices during training.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec


class WeightVisualizer:
    """Visualize neural network weights in real-time during training."""
    
    def __init__(self, network, update_interval=10):
        """
        Initialize visualizer.
        
        Args:
            network: NeuralNetwork instance to visualize
            update_interval: Update visualization every N batches
        """
        self.network = network
        self.update_interval = update_interval
        self.batch_count = 0
        
        # Training metrics
        self.losses = []
        self.accuracies = []
        self.batch_numbers = []
        
        # Set up the figure
        self.setup_figure()
        
    def setup_figure(self):
        """Create the visualization figure with subplots."""
        plt.ion()  # Interactive mode
        
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('MNIST Neural Network - Live Weight Visualization', 
                         fontsize=14, fontweight='bold')
        
        # Create grid for subplots
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.30, wspace=0.25)
        
        # Weight heatmaps
        self.ax_weights1 = self.fig.add_subplot(gs[0, 0])
        self.ax_weights2 = self.fig.add_subplot(gs[0, 1])
        
        # Loss and accuracy plots
        self.ax_loss = self.fig.add_subplot(gs[1, :])
        self.ax_accuracy = self.fig.add_subplot(gs[2, :])
        
        # Initialize weight heatmaps
        self._init_weight_heatmaps()
        
        # Initialize metric plots
        self._init_metric_plots()
        
        plt.show(block=False)
        plt.pause(0.001)
    
    def _init_weight_heatmaps(self):
        """Initialize weight heatmap visualizations."""
        # Hidden layer 1: 784 -> 128
        weights1 = self.network.weights[0]
        # Reshape to approximate square for better visualization
        # Take first 256 input neurons (16x16) to all 128 hidden neurons
        w1_subset = weights1[:256, :].T  # Shape: (128, 256)
        
        self.im1 = self.ax_weights1.imshow(w1_subset, cmap='RdBu', aspect='auto',
                                           vmin=-1, vmax=1, interpolation='nearest')
        self.ax_weights1.set_title('Hidden Layer 1 Weights (128 neurons × 256 inputs)', 
                                   fontsize=10, fontweight='bold')
        self.ax_weights1.set_xlabel('Input Neuron (subset)', fontsize=9)
        self.ax_weights1.set_ylabel('Hidden Layer 1 Neuron', fontsize=9)
        self.fig.colorbar(self.im1, ax=self.ax_weights1, label='Weight Value')
        
        # Hidden layer 2: 128 -> 64
        weights2 = self.network.weights[1]
        self.im2 = self.ax_weights2.imshow(weights2.T, cmap='RdBu', aspect='auto',
                                           vmin=-1, vmax=1, interpolation='nearest')
        self.ax_weights2.set_title('Hidden Layer 2 Weights (64 neurons × 128 inputs)', 
                                   fontsize=10, fontweight='bold')
        self.ax_weights2.set_xlabel('Hidden Layer 1 Neuron', fontsize=9)
        self.ax_weights2.set_ylabel('Hidden Layer 2 Neuron', fontsize=9)
        self.fig.colorbar(self.im2, ax=self.ax_weights2, label='Weight Value')
    
    def _init_metric_plots(self):
        """Initialize loss and accuracy plots."""
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='Training Loss')
        self.ax_loss.set_xlabel('Batch Number', fontsize=9, fontweight='bold')
        self.ax_loss.set_ylabel('Cross-Entropy Loss', fontsize=9, fontweight='bold')
        self.ax_loss.set_title('Training Loss Over Time', fontsize=10, fontweight='bold')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend(fontsize=8)
        
        self.line_acc, = self.ax_accuracy.plot([], [], 'g-', linewidth=2, label='Training Accuracy')
        self.ax_accuracy.set_xlabel('Batch Number', fontsize=9, fontweight='bold')
        self.ax_accuracy.set_ylabel('Accuracy (%)', fontsize=9, fontweight='bold')
        self.ax_accuracy.set_title('Training Accuracy Over Time', fontsize=10, fontweight='bold')
        self.ax_accuracy.grid(True, alpha=0.3)
        self.ax_accuracy.legend(fontsize=8)
        self.ax_accuracy.set_ylim([0, 105])
    
    def update(self, batch_num, loss, accuracy=None):
        """
        Update visualization with new training metrics.
        
        Args:
            batch_num: Current batch number
            loss: Loss value for current batch
            accuracy: Accuracy value (optional)
        """
        self.batch_count += 1
        
        # Store metrics
        self.batch_numbers.append(batch_num)
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy * 100)
        
        # Update only every N batches
        if self.batch_count % self.update_interval != 0:
            return
        
        # Update weight heatmaps
        self._update_weight_heatmaps()
        
        # Update loss plot
        self.line_loss.set_data(self.batch_numbers, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # Update accuracy plot
        if self.accuracies:
            self.line_acc.set_data(self.batch_numbers, self.accuracies)
            self.ax_accuracy.relim()
            self.ax_accuracy.autoscale_view()
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _update_weight_heatmaps(self):
        """Update weight heatmap visualizations."""
        # Update hidden layer 1 weights
        weights1 = self.network.weights[0]
        w1_subset = weights1[:256, :].T
        
        # Dynamically adjust color scale based on current weight range
        vmin1, vmax1 = np.percentile(w1_subset, [5, 95])
        vmin1, vmax1 = max(vmin1, -2), min(vmax1, 2)
        self.im1.set_data(w1_subset)
        self.im1.set_clim(vmin=vmin1, vmax=vmax1)
        
        # Update hidden layer 2 weights
        weights2 = self.network.weights[1]
        vmin2, vmax2 = np.percentile(weights2, [5, 95])
        vmin2, vmax2 = max(vmin2, -2), min(vmax2, 2)
        self.im2.set_data(weights2.T)
        self.im2.set_clim(vmin=vmin2, vmax=vmax2)
    
    def show_sample_predictions(self, X_sample, y_true, num_samples=10):
        """
        Display sample predictions with images.
        
        Args:
            X_sample: Sample input images (num_samples, 784)
            y_true: True labels (num_samples,)
            num_samples: Number of samples to display
        """
        # Make predictions
        y_pred = self.network.predict(X_sample[:num_samples])
        
        # Create figure for predictions
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Reshape to 28x28
                image = X_sample[i].reshape(28, 28)
                ax.imshow(image, cmap='gray')
                
                # Color code correct/incorrect
                color = 'green' if y_pred[i] == y_true[i] else 'red'
                ax.set_title(f'Pred: {y_pred[i]}, True: {y_true[i]}', 
                           color=color, fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)
