"""
MNIST Training Script with Live Weight Visualization
Train a neural network from scratch and visualize weight updates in real-time.
"""

import numpy as np
from mnist_classifier import NeuralNetwork
from data_loader import MNISTLoader
from weight_visualizer import WeightVisualizer
import time


def train_mnist(epochs=10, batch_size=128, learning_rate=0.01, 
                visualization_update_interval=10):
    """
    Train MNIST digit classifier with live visualization.
    
    Args:
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate for gradient descent
        visualization_update_interval: Update visualization every N batches
    """
    print("="*60)
    print("MNIST Digit Classifier - Training with Live Visualization")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading MNIST dataset...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load_data(normalize=True)
    
    # Create validation split (use last 10% of training data)
    val_split = int(0.9 * len(X_train))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize neural network
    print("\n[2/4] Initializing neural network...")
    network = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=learning_rate
    )
    print(f"Architecture: {' -> '.join(map(str, network.layer_sizes))}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    
    # Initialize visualizer
    print("\n[3/4] Setting up live visualization...")
    visualizer = WeightVisualizer(network, update_interval=visualization_update_interval)
    print("Visualization ready! Watch the weights light up during training.")
    
    # Training loop
    print("\n[4/4] Starting training...")
    print("-" * 60)
    
    best_val_accuracy = 0.0
    batch_num = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Create mini-batches
        batch_generator = loader.create_mini_batches(
            X_train, y_train, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Train on each batch
        for batch_idx, (X_batch, y_batch_onehot) in enumerate(batch_generator):
            # Training step
            loss = network.train_step(X_batch, y_batch_onehot)
            epoch_losses.append(loss)
            batch_num += 1
            
            # Calculate batch accuracy for visualization
            y_batch_pred = network.predict(X_batch)
            y_batch_true = np.argmax(y_batch_onehot, axis=1)
            batch_accuracy = np.mean(y_batch_pred == y_batch_true)
            
            # Update visualization
            visualizer.update(batch_num, loss, batch_accuracy)
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss = {loss:.4f}, Acc = {batch_accuracy*100:.2f}%")
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        
        # Evaluate on validation set
        val_accuracy = network.evaluate(X_val, y_val)
        
        print(f"\n  Epoch Summary:")
        print(f"    Average Loss: {avg_loss:.4f}")
        print(f"    Validation Accuracy: {val_accuracy*100:.2f}%")
        print(f"    Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            network.save('best_model.pkl')
            print(f"    ✓ New best model saved! (Validation Accuracy: {val_accuracy*100:.2f}%)")
        
        print("-" * 60)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    test_accuracy = network.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Best Validation Accuracy: {best_val_accuracy*100:.2f}%")
    
    # Show sample predictions
    print("\nGenerating sample predictions...")
    visualizer.show_sample_predictions(X_test, y_test, num_samples=10)
    
    # Keep visualization open
    print("\n✓ Training complete! Close the visualization window to exit.")
    input("Press Enter to close...")
    visualizer.close()


if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    VISUALIZATION_UPDATE_INTERVAL = 5  # Update every 5 batches
    
    # Start training
    train_mnist(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        visualization_update_interval=VISUALIZATION_UPDATE_INTERVAL
    )
