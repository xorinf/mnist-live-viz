"""
MNIST Data Loader
Handles downloading, parsing, and preprocessing MNIST dataset.
"""

import numpy as np
import gzip
import os
import urllib.request


class MNISTLoader:
    """Load and preprocess MNIST dataset."""
    
    # Use GitHub mirror instead of original URL (which has issues)
    BASE_URL = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    def __init__(self, data_dir='./mnist_data'):
        """
        Initialize MNIST loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def download(self):
        """Download MNIST dataset if not already present."""
        for key, filename in self.FILES.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                url = self.BASE_URL + filename
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
            else:
                print(f"{filename} already exists")
    
    def _read_images(self, filepath):
        """
        Read MNIST image file in IDX format.
        
        Args:
            filepath: Path to gzipped IDX file
            
        Returns:
            numpy array of images (num_images, 784)
        """
        with gzip.open(filepath, 'rb') as f:
            # Read header
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, rows * cols)
            
        return data
    
    def _read_labels(self, filepath):
        """
        Read MNIST label file in IDX format.
        
        Args:
            filepath: Path to gzipped IDX file
            
        Returns:
            numpy array of labels (num_labels,)
        """
        with gzip.open(filepath, 'rb') as f:
            # Read header
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # Read label data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            
        return data
    
    def load_data(self, normalize=True):
        """
        Load and preprocess MNIST data.
        
        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Download if needed
        self.download()
        
        # Load training data
        train_images_path = os.path.join(self.data_dir, self.FILES['train_images'])
        train_labels_path = os.path.join(self.data_dir, self.FILES['train_labels'])
        self.X_train = self._read_images(train_images_path)
        self.y_train = self._read_labels(train_labels_path)
        
        # Load test data
        test_images_path = os.path.join(self.data_dir, self.FILES['test_images'])
        test_labels_path = os.path.join(self.data_dir, self.FILES['test_labels'])
        self.X_test = self._read_images(test_images_path)
        self.y_test = self._read_labels(test_labels_path)
        
        # Normalize pixel values
        if normalize:
            self.X_train = self.X_train.astype(np.float32) / 255.0
            self.X_test = self.X_test.astype(np.float32) / 255.0
        
        print(f"Training set: {self.X_train.shape[0]} images")
        print(f"Test set: {self.X_test.shape[0]} images")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def create_mini_batches(self, X, y, batch_size=32, shuffle=True):
        """
        Create mini-batches for training.
        
        Args:
            X: Input data (num_samples, 784)
            y: Labels (num_samples,)
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle data before creating batches
            
        Yields:
            Tuple of (X_batch, y_batch_onehot)
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Convert labels to one-hot encoding
            y_batch_onehot = self.to_one_hot(y_batch)
            
            yield X_batch, y_batch_onehot
    
    def to_one_hot(self, y, num_classes=10):
        """
        Convert integer labels to one-hot encoding.
        
        Args:
            y: Integer labels (num_samples,)
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels (num_samples, num_classes)
        """
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
