import io
import sys
import os
import cv2
import numpy as np
import torch
import psutil
import math
import random
import copy
import time
import pickle
from scipy.signal import correlate, convolve
from matplotlib import image as mpimg
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import trange
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
import seaborn as sns  
from sklearn.manifold import TSNE  
from sklearn.decomposition import PCA  
import warnings

DTYPE = np.float32

def plot_results(losses: dict, accuracies: dict):
    plt.figure(figsize=(18, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses["train"], label="Train Loss")
    plt.plot(losses["val"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies["train"], label="Train Accuracy")
    plt.plot(accuracies["val"], label="Val Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("result_train_iter3.png", dpi=300)


def plot_confusion_matrix(y_true, y_pred, class_names, kept_classes):
    dim = len(kept_classes)
    labels = [class_names[i] for i in kept_classes]
    # Plot the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    norm_conf_mat = conf_mat / np.sum(conf_mat, axis=1)
    # Plot the matrix
    fig, ax = plt.subplots()
    plt.imshow(norm_conf_mat)
    plt.title('Confusion Matrix')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.xticks(range(dim), labels, rotation=45)
    plt.yticks(range(dim), labels)
    plt.colorbar()
    # Put number of each cell in plot
    for i in range(dim):
        for j in range(dim):
            c = conf_mat[j, i]
            color = 'black' if c > 500 else 'white'
            ax.text(i, j, str(int(c)), va='center', ha='center', color=color)
    plt.savefig("confusion_matrix_iter3.png", dpi=300)

def onehot_encoder(y, num_labels):
    one_hot = np.zeros(shape=(y.size, num_labels), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot

class Layer:
    def __init__(self):
        self.inp = None
        self.out = None

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        pass

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.w = 0.1 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(DTYPE)
        self.b = np.zeros((out_channels, 1), dtype=DTYPE)
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        batch_size, in_channels, height, width = inp.shape

        # Padding
        self.padded_inp = np.pad(inp, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Apply im2col
        strides = (
            self.padded_inp.strides[0],
            self.padded_inp.strides[1],
            self.stride * self.padded_inp.strides[2],
            self.stride * self.padded_inp.strides[3],
            self.padded_inp.strides[2],
            self.padded_inp.strides[3]
        )
        shape = (batch_size, in_channels, out_height, out_width, self.kernel_size, self.kernel_size)
        col = as_strided(
            self.padded_inp,
            shape=shape,
            strides=strides
        )
        col = col.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_height * out_width, -1)

        # Reshape kernel before dot
        w_col = self.w.reshape(self.out_channels, -1).T

        # Calculate output
        out = np.dot(col, w_col) + self.b.T
        out = out.reshape(batch_size, out_height, out_width, self.out_channels).transpose(0, 3, 1, 2)

        return out
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        batch_size, in_channels, height, width = self.inp.shape
        _, _, out_height, out_width = up_grad.shape

        # Pad input again
        self.padded_inp = np.pad(self.inp, ((0, 0), (0, 0), 
                                            (self.padding, self.padding), 
                                            (self.padding, self.padding)), mode='constant')

        # Create im2col for input
        strides = (
            self.padded_inp.strides[0],
            self.padded_inp.strides[1],
            self.stride * self.padded_inp.strides[2],
            self.stride * self.padded_inp.strides[3],
            self.padded_inp.strides[2],
            self.padded_inp.strides[3]
        )
        shape = (batch_size, in_channels, out_height, out_width, self.kernel_size, self.kernel_size)
        col = np.lib.stride_tricks.as_strided(self.padded_inp, shape=shape, strides=strides)
        col = col.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_height * out_width, -1)  # [N*out_h*out_w, C*k*k]

        # Reshape up_grad to dY_col
        dY = up_grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)  # [N*out_h*out_w, C_out]

        # 3. Calculate ∂L/∂w
        self.dw = dY.T @ col  # [C_out, C_in*k*k]
        self.dw = self.dw.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # 4. Calculate ∂L/∂b
        self.db = np.sum(dY, axis=0).reshape(self.out_channels, 1)

        # 5. Calculate ∂L/∂x over col2im
        w_flat = self.w.reshape(self.out_channels, -1)  # [C_out, C_in*k*k]
        dX_col = dY @ w_flat  # [N*out_h*out_w, C_in*k*k]
        dX_col = dX_col.reshape(batch_size, out_height, out_width, in_channels, self.kernel_size, self.kernel_size)
        dX_col = dX_col.transpose(0, 3, 4, 5, 1, 2)

        # Calculte output (padded)
        dX_padded = np.zeros_like(self.padded_inp)

        for i in range(out_height):
            for j in range(out_width):
                dX_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += dX_col[:, :, :, :, i, j]

        # Remove padding 
        if self.padding > 0:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return dX_padded
    def step(self, lr: float) -> None:
        """Update weights and biases."""
        self.w -= lr * self.dw
        self.b -= lr * self.db

class GlobalAvgPool2D(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp_shape = inp.shape  # Save input shape for backward pass
        return np.mean(inp, axis=(2, 3))  # Output shape: (batch, channels)

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """
        Evenly distribute the gradient across all elements in the pooling region.
        """
        batch_size, channels = up_grad.shape
        height, width = self.inp_shape[2], self.inp_shape[3]

        # Each output element corresponds to (height * width) input elements to average gradient
        down_grad = up_grad[:, :, np.newaxis, np.newaxis] / (height * width)

        # Repeat the gradient to match the original input shape
        return np.tile(down_grad, (1, 1, height, width))

    
class MaxPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass of max pooling with full vectorization."""
        self.inp = inp
        batch_size, channels, height, width = inp.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        shape = (batch_size, channels, out_height, out_width, self.pool_size, self.pool_size)
        strides = (
            inp.strides[0],
            inp.strides[1],
            inp.strides[2] * self.stride,
            inp.strides[3] * self.stride,
            inp.strides[2],
            inp.strides[3]
        )

        self.patches = np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)
        self.mask = (self.patches == np.max(self.patches, axis=(4, 5), keepdims=True))
        out = np.max(self.patches, axis=(4, 5))
        return out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for max pooling with full vectorization."""
        batch_size, channels, height, width = self.inp.shape
        out_height, out_width = up_grad.shape[2], up_grad.shape[3]

        # Expand gradient to match the mask shape
        grad_expanded = up_grad[:, :, :, :, None, None]
        grad_broadcasted = grad_expanded * self.mask

        down_grad = np.zeros_like(self.inp)

        for i in range(self.pool_size):
            for j in range(self.pool_size):
                down_grad[:, :, 
                          i:self.stride*out_height+i:self.stride, 
                          j:self.stride*out_width+j:self.stride] += grad_broadcasted[:, :, :, :, i, j]

        return down_grad
    
class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
        self.training = True  # Will be set by model.train()

    def forward(self, x):
        if self.training:
            # Generate a binary mask where elements are dropped with probability 'rate'
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(x.dtype)
            # Scale the output to maintain the same expected value
            return x * self.mask / (1.0 - self.rate)
        else:
            # No dropout applied during evaluation
            return x

    def backward(self, d_out):
        if self.training:
            # Apply the same mask to the gradient and scale
            return d_out * self.mask / (1.0 - self.rate)
        else:
            # No modification to the gradient during evaluation
            return d_out


class AvgPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Average Pooling Layer.

        Args:
            pool_size (int): Size of the pooling window.
            stride (int): Step size for moving the pooling window.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """
        Forward pass of average pooling.

        Args:
            inp (np.ndarray): Input tensor of shape (batch, channels, height, width).

        Returns:
            np.ndarray: Output tensor after average pooling.
        """
        self.inp = inp
        batch_size, channels, height, width = inp.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # Initialize output array
        out = np.zeros((batch_size, channels, out_height, out_width), dtype=DTYPE)

        # Compute pooling window starting indices
        i_idx = np.arange(0, height - self.pool_size + 1, self.stride)
        j_idx = np.arange(0, width - self.pool_size + 1, self.stride)

        # Generate 2D grids of indices
        i_grid, j_grid = np.meshgrid(i_idx, j_idx, indexing='ij')

        # Extract pooling regions
        region = inp[:, :, i_grid[:, :, None], j_grid[:, None, :]]

        # Compute mean over each pooling region
        out = np.mean(region, axis=(2, 3))

        self.out = out
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of average pooling.

        Args:
            up_grad (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        batch_size, channels, height, width = self.inp.shape
        down_grad = np.zeros_like(self.inp)

        out_height, out_width = up_grad.shape[2], up_grad.shape[3]

        # Compute pooling window starting indices
        i_idx = np.arange(0, height - self.pool_size + 1, self.stride)
        j_idx = np.arange(0, width - self.pool_size + 1, self.stride)
        i_grid, j_grid = np.meshgrid(i_idx, j_idx, indexing='ij')

        # Distribute gradient equally across each pooling region
        region_grad = up_grad[:, :, :, :, None, None] / (self.pool_size * self.pool_size)

        # Accumulate gradients to corresponding input positions
        np.add.at(down_grad, (slice(None), slice(None), i_grid[:, :, None], j_grid[:, None, :]), region_grad)

        return down_grad


class Flatten(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Flatten the input into a 2D array."""
        self.inp_shape = inp.shape
        return inp.reshape(self.inp_shape[0], -1)

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Reshape the gradient back to the original input shape."""
        return up_grad.reshape(self.inp_shape)

class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # He initialization: better scaling for deep networks
        self.w = 0.1 * np.random.randn(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Perform the linear transformation: output = inp * W + b"""
        self.inp = inp
        self.out = np.dot(inp, self.w) + self.b
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backpropagate the gradients through this layer."""
        # Compute gradients for weights and biases
        self.dw = np.dot(self.inp.T, up_grad)  # Gradient wrt weights
        self.db = np.sum(up_grad, axis=0, keepdims=True)  # Gradient wrt biases
        # Compute gradient to propagate back (downstream)
        down_grad = np.dot(up_grad, self.w.T)
        return down_grad

    def step(self, lr: float) -> None:
        """Update the weights and biases using the gradients."""
        self.w -= lr * self.dw
        self.b -= lr * self.db


class ReLU(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """ReLU Activation: f(x) = max(0, x)"""
        self.inp = inp
        self.out = np.maximum(0, inp)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU: derivative is 1 where input > 0, else 0."""
        down_grad = up_grad * (self.inp > 0)  # Efficient boolean indexing
        return down_grad


class Softmax(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Softmax Activation: f(x) = exp(x) / sum(exp(x))"""
        # Subtract max for numerical stability
        exp_values = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for Softmax using the Jacobian matrix."""
        down_grad = np.empty_like(up_grad)
        for i in range(up_grad.shape[0]):
            single_output = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            down_grad[i] = np.dot(jacobian, up_grad[i])
        return down_grad

class Loss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.loss = None

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

class CrossEntropy(Loss):
    """
    Custom implementation of the cross-entropy loss function.
    Suitable for multi-class classification with one-hot encoded targets.
    """

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass for computing cross-entropy loss.

        Args:
            prediction (np.ndarray): Predicted probabilities (after softmax), shape (batch_size, num_classes).
            target (np.ndarray): One-hot encoded ground truth labels, shape (batch_size, num_classes).

        Returns:
            float: Mean cross-entropy loss over the batch.
        """
        self.prediction = prediction
        self.target = target

        # Clip predictions to prevent log(0)
        self.clipped_pred = np.clip(prediction, 1e-12, 1.0)
        self.loss = -np.mean(np.sum(target * np.log(self.clipped_pred), axis=1))
        return self.loss

    def backward(self) -> np.ndarray:
        """
        Backward pass for cross-entropy loss.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input predictions.
        """
        grad = -self.target / self.clipped_pred / self.target.shape[0]
        return grad


class CNN:
    """
    Custom implementation of a Convolutional Neural Network (CNN).
    Supports training, evaluation, saving/loading weights, and backpropagation.
    """

    def __init__(self, layers: list[Layer], loss_fn: Loss, lr: float) -> None:
        """
        Initialize the CNN.

        Args:
            layers (list[Layer]): List of layers in the network.
            loss_fn (Loss): Loss function used for training.
            lr (float): Learning rate.
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr = lr
        self.best_val_acc = 0.0
        self.best_weights = None

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Allow the model to be called like a function."""
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute the loss."""
        return self.loss_fn(prediction, target)

    def backward(self) -> None:
        """Backward pass to propagate gradients through layers."""
        up_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            up_grad = layer.backward(up_grad)

    def update(self) -> None:
        """Update layer parameters using their gradients."""
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(self.lr)

    def train(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_val: np.ndarray, 
        y_val: np.ndarray, 
        epochs: int, 
        batch_size: int, 
        kept_classes: list
    ) -> tuple:
        """
        Train the model and save the best weights based on validation accuracy.

        Returns:
            tuple: Dictionaries of losses and accuracies for train and validation sets.
        """
        losses = {"train": [], "val": []}
        accuracies = {"train": [], "val": []}
        self.best_val_acc = 0.0
        self.best_epoch = -1
        best_weights = None

        for epoch in (pbar := trange(epochs)):
            epoch_start = time.time()

            # ========== Training ==========
            running_loss = 0.0
            correct = 0
            forward_total, backward_total, update_total = 0, 0, 0

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                y_batch_onehot = onehot_encoder(y_batch, len(kept_classes))

                start = time.time()
                prediction = self.forward(x_batch)
                forward_total += time.time() - start

                running_loss += self.loss(prediction, y_batch_onehot) * x_batch.shape[0]
                correct += np.sum(np.argmax(prediction, axis=1) == y_batch)

                start = time.time()
                self.backward()
                backward_total += time.time() - start

                start = time.time()
                self.update()
                update_total += time.time() - start

            train_loss = running_loss / len(x_train)
            train_acc = 100 * correct / len(x_train)
            losses["train"].append(train_loss)
            accuracies["train"].append(train_acc)

            print(f"Forward: {forward_total:.4f}s | Backward: {backward_total:.4f}s | Update: {update_total:.4f}s")

            # ========== Validation ==========
            val_loss, val_acc = self.evaluate(x_val, y_val, batch_size, kept_classes)
            losses["val"].append(val_loss)
            accuracies["val"].append(val_acc)

            # ========== Save Best ==========
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                best_weights = [
                    copy.deepcopy(layer.w) if isinstance(layer, (Conv2D, Linear)) else None
                    for layer in self.layers
                ]
                new_best_note = " | *** New Best ***"
            else:
                new_best_note = ""

            # Logging
            epoch_time = time.time() - epoch_start
            pbar.set_description(f"Epoch {epoch + 1:02d}")
            pbar.write(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s{new_best_note}"
            )

        # Load best weights after training
        for layer, weights in zip(self.layers, best_weights):
            if isinstance(layer, (Conv2D, Linear)) and weights is not None:
                layer.w = weights

        print(f"\nBest model found at epoch {self.best_epoch} with Val Acc: {self.best_val_acc:.2f}%")
        return losses, accuracies

    def evaluate(self, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, kept_classes: list) -> tuple:
        """
        Evaluate the model on the validation set.

        Returns:
            tuple: Validation loss and accuracy.
        """
        val_loss = 0.0
        correct = 0
        for i in range(0, len(x_val), batch_size):
            x_batch = x_val[i:i + batch_size]
            y_batch = y_val[i:i + batch_size]
            y_batch_onehot = onehot_encoder(y_batch, num_labels=len(kept_classes))
            prediction = self.forward(x_batch)
            val_loss += self.loss(prediction, y_batch_onehot) * batch_size
            correct += np.sum(np.argmax(prediction, axis=1) == y_batch)

        val_loss /= len(x_val)
        val_acc = 100 * correct / len(x_val)
        return val_loss, val_acc

    def save_model(self, filepath: str):
        """
        Save model weights to an .npz file.

        Args:
            filepath (str): Path to the file.
        """
        save_dict = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Linear)):
                save_dict[f"layer{idx}_w"] = layer.w
                if hasattr(layer, 'b') and layer.b is not None:
                    save_dict[f"layer{idx}_b"] = layer.b
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model weights from an .npz file.

        Args:
            filepath (str): Path to the saved file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' does not exist.")
        data = np.load(filepath)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Linear)):
                if f"layer{idx}_w" in data:
                    layer.w = data[f"layer{idx}_w"]
                if f"layer{idx}_b" in data:
                    layer.b = data[f"layer{idx}_b"]
        print(f"Model weights loaded from {filepath}")

    def save_weights(self, filename: str):
        """Save trainable layer weights using pickle."""
        weights = []
        for layer in self.layers:
            if isinstance(layer, (Conv2D, Linear)):
                weights.append({'w': copy.deepcopy(layer.w), 'b': copy.deepcopy(layer.b)})
            else:
                weights.append(None)
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filename: str):
        """Load weights from a pickle file."""
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if isinstance(layer, (Conv2D, Linear)):
                layer.w = weights[idx]['w']
                layer.b = weights[idx]['b']
                idx += 1

    
def rotate_image(image, angle):
    """
    Rotate the input image by a specified angle.

    Args:
        image (ndarray): Input image in (H, W, C) format.
        angle (float): Angle to rotate (in degrees).

    Returns:
        ndarray: Rotated image.
    """
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def load_images_from_folder(folder, label, image_size=(64, 64), augment=False):
    """
    Load images from a specified folder, apply resizing and normalization.
    Optionally apply rotation-based augmentation.

    Args:
        folder (str): Path to the image folder.
        label (int): Integer label associated with the folder/class.
        image_size (tuple): Desired image size (width, height).
        augment (bool): If True, apply rotation augmentations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of images and corresponding labels.
    """
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize and normalize the image
            img = cv2.resize(img, image_size)
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)
            images.append(img)
            labels.append(label)
            
            if augment:
                # Apply multiple rotation angles
                angles = [15, 30, -15, -30]
                for angle in angles:
                    rotated_img = rotate_image(img.transpose(1, 2, 0), angle)  # Convert to (H, W, C)
                    rotated_img = np.transpose(rotated_img, (2, 0, 1))  # Convert back to (C, H, W)
                    images.append(rotated_img)
                    labels.append(label)
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def visualize_features(model, x_data, y_data, num_samples=1000):
    """
    Extract features from intermediate CNN layers and visualize using PCA and t-SNE.

    Args:
        model: Trained CNN model with custom layers.
        x_data (np.ndarray): Input data (N, C, H, W).
        y_data (np.ndarray): Corresponding labels.
        num_samples (int): Number of samples to visualize (default: 1000).

    Saves:
        A side-by-side PCA and t-SNE plot as 'visualization_feature_iter3.png'.
    """
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
    
    # Extract features from the network up to (but not including) GlobalAvgPool2D
    current_input = x_data.copy()
    for layer in model.layers:
        if isinstance(layer, GlobalAvgPool2D):
            break
        current_input = layer.forward(current_input)
    features = current_input

    # Limit the number of samples for visualization
    num_samples = min(num_samples, len(features))
    indices = np.random.choice(len(features), num_samples, replace=False)
    sampled_features = features[indices]
    sampled_labels = y_data[indices]

    # Flatten the features to 2D: (N, C*H*W)
    if isinstance(sampled_features, torch.Tensor):
        sampled_features = sampled_features.detach().cpu().numpy()
    sampled_features_flat = sampled_features.reshape(sampled_features.shape[0], -1)

    # Apply PCA to reduce dimensionality before t-SNE
    max_pca_components = min(sampled_features_flat.shape[0], sampled_features_flat.shape[1])
    n_components = min(50, max_pca_components)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(sampled_features_flat)

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(pca_result)

    # Plot both PCA and t-SNE visualizations
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=sampled_labels, cmap="viridis", alpha=0.6)
    plt.title("PCA Visualization")

    plt.subplot(1, 2, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=sampled_labels, cmap="viridis", alpha=0.6)
    plt.title("t-SNE Visualization")

    plt.savefig("visualization_feature_iter3.png", dpi=300)


def visualize_layer_output(model, img):
    """
    Visualize intermediate feature maps of a CNN layer by layer.

    Args:
        model: Trained CNN model.
        img (ndarray): Input image in (H, W, C) format.

    Displays:
        The output of each Conv2D and MaxPool2D layer using matplotlib.
    """
    # Prepare the image
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # Convert to (1, C, H, W)

    outputs = []
    current_input = img

    for layer in model.layers:
        current_input = layer.forward(current_input)
        if isinstance(layer, (Conv2D, MaxPool2D)):
            outputs.append(current_input)

    # Visualize the first channel of each output
    for i, out in enumerate(outputs):
        plt.figure(figsize=(10, 3))
        plt.title(f"Layer {i+1} Output")
        plt.imshow(out[0, 0], cmap="viridis")
        plt.axis('off')
        plt.show()


def main():
    """
    Train a custom CNN on the brain tumor MRI dataset with 4 classes: 
    Normal, Meningioma, Glioma, and Pituitary.
    Includes image augmentation, data shuffling, training, evaluation, and visualization.
    """

    # Load "Normal" images (label=0) with augmentation
    normal_images, normal_labels = load_images_from_folder(
        'C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Training/normal',
        label=0,
        augment=True
    )

    # Load "Meningioma" images (label=1) with augmentation
    meningioma_images, meningioma_labels = load_images_from_folder(
        'C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Training/meningioma',
        label=1,
        augment=True
    )

    # Load "Glioma" images (label=2) with augmentation
    glioma_images, glioma_labels = load_images_from_folder(
        'C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Training/glioma',
        label=2,
        augment=True
    )

    # Load "Pituitary" images (label=3) with augmentation
    pituitary_images, pituitary_labels = load_images_from_folder(
        'C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Training/pituitary',
        label=3,
        augment=True
    )

    # Combine all images and labels
    X = np.concatenate([normal_images, meningioma_images, glioma_images, pituitary_images], axis=0)
    y = np.concatenate([normal_labels, meningioma_labels, glioma_labels, pituitary_labels], axis=0)

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split into training and validation sets (80/20 split)
    split_idx = int(0.8 * len(X))
    x_train, x_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Define the list of class indices to keep (0 = Normal, ..., 3 = Pituitary)
    kept_classes = [0, 1, 2, 3]

    # Define the CNN architecture
    layers = [
        # Block 1
        Conv2D(3, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        # Block 2
        Conv2D(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        # Block 3
        Conv2D(64, 128, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        # Block 4
        Conv2D(128, 256, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),  # Additional pooling for size reduction

        # Block 5 (newly added)
        Conv2D(256, 512, kernel_size=3, padding=1),
        ReLU(),
        GlobalAvgPool2D(),  # Output shape: (batch_size, 512)

        # Fully connected layers
        Flatten(),
        Linear(512, 128),
        ReLU(),
        Dropout(rate=0.5),  # Helps prevent overfitting
        Linear(128, len(kept_classes)),
        Softmax()
    ]

    # Initialize and train the model
    model = CNN(layers, CrossEntropy(), lr=0.01)
    losses, accuracies = model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=50,
        batch_size=16,
        kept_classes=kept_classes
    )

    print(f"Best Validation Accuracy: {model.best_val_acc:.2f}%")

    # Save trained model weights
    model.save_model("model_weights_iter3.npz")

    # Predict on validation set
    y_pred = np.argmax(model.forward(x_val), axis=1)

    # Plot confusion matrix
    plot_confusion_matrix(
        y_val, y_pred,
        class_names=["Normal", "Meningioma", "Glioma", "Pituitary"],
        kept_classes=kept_classes
    )

    # Visualize feature distribution using PCA + t-SNE
    visualize_features(model, x_val, y_val)

    # Plot training loss and accuracy curves
    plot_results(losses, accuracies)

# Run the main function
if __name__ == "__main__":
    main()
