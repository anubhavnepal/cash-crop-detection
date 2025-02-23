# -------------------
# Activation Functions
# -------------------

def relu(x):
    return cp.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = cp.exp(x - cp.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exps / cp.sum(exps, axis=-1, keepdims=True)


# -------------------
# Layers
# -------------------
class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.input > 0).astype(float)
        
class Conv2D:
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = None
        self.biases = None
         # Storing the output for backward pass
        self.output = None 

    def initialize(self, input_shape):
        self.input_shape = input_shape
        fan_in = self.kernel_size * self.kernel_size * input_shape[-1]
        scale = cp.sqrt(2.0 / fan_in)  # He initialization
        self.filters = cp.random.randn(self.num_filters, self.kernel_size, self.kernel_size, input_shape[-1]) * scale
        self.biases = cp.zeros(self.num_filters)

    def forward(self, input_data):
        self.input_data = input_data
        if self.filters is None:
            # Initializing filters based on input shape
            self.initialize(input_data.shape[1:])  
        
        pad_width = ((0, 0), (self.kernel_size // 2, self.kernel_size // 2),
                     (self.kernel_size // 2, self.kernel_size // 2), (0, 0))
        padded_input = cp.pad(input_data, pad_width, mode='constant')
        windows = cp.lib.stride_tricks.sliding_window_view(padded_input, (self.kernel_size, self.kernel_size), axis=(1, 2))
        windows = windows.reshape(-1, self.kernel_size, self.kernel_size, input_data.shape[-1])
        output = cp.tensordot(windows, self.filters, axes=([1,2,3], [1,2,3]))
        output = output.reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2], self.num_filters)
        output += self.biases
        self.output = output
        return output

    def backward(self, d_out, learning_rate=0.001, l2_reg=1e-4):
        d_out = d_out * relu_derivative(self.output)
        batch_size, h_out, w_out, num_filters = d_out.shape
        
        d_out_reshaped = d_out.transpose(0, 3, 1, 2).reshape(-1, num_filters)
        
        pad_width = ((0, 0), (self.kernel_size//2, self.kernel_size//2),
                     (self.kernel_size//2, self.kernel_size//2), (0, 0))
        padded_input = cp.pad(self.input_data, pad_width, mode='constant')
        windows = cp.lib.stride_tricks.sliding_window_view(
            padded_input, (self.kernel_size, self.kernel_size), axis=(1, 2)
        ).reshape(-1, self.kernel_size, self.kernel_size, self.input_data.shape[-1])
        
        # Computing gradient for filters
        d_filters = cp.dot(windows.T, d_out_reshaped).T.reshape(
            self.num_filters, self.kernel_size, self.kernel_size, self.input_data.shape[-1]
        )
       
       # Adding after gradient calculation
        d_filters = cp.clip(d_filters, -1, 1)  

        # Adding L2 regularization
        d_filters += l2_reg * self.filters
        
        # Computing gradient for biases
        d_biases = cp.sum(d_out, axis=(0, 1, 2))
        
        # Computing gradient for input
        padded_d_out = cp.pad(d_out, 
            ((0, 0), (self.kernel_size//2, self.kernel_size//2),
             (self.kernel_size//2, self.kernel_size//2), (0, 0)),
            mode='constant'
        )
        flipped_filters = cp.flip(self.filters, axis=(1, 2))
        input_windows = cp.lib.stride_tricks.sliding_window_view(
            padded_d_out, (self.kernel_size, self.kernel_size), axis=(1, 2)
        )
        input_windows = input_windows.reshape(-1, self.kernel_size, self.kernel_size, num_filters)
        d_input = cp.tensordot(input_windows, flipped_filters, axes=([1, 2, 3], [1, 2, 0]))
        d_input = d_input.reshape(
            padded_d_out.shape[0],
            padded_d_out.shape[1] - self.kernel_size + 1,
            padded_d_out.shape[2] - self.kernel_size + 1,
            self.input_data.shape[-1]
        )
        
        # Updating parameters
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        
        return d_input


class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
         # Storing mask for backward pass
        self.mask = None

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, h, w, c = input_data.shape
        
        # Ensuring dimensions are divisible by pool_size
        if h % self.pool_size != 0 or w % self.pool_size != 0:
            raise ValueError(f"Input dimensions ({h}, {w}) are not divisible by pool_size {self.pool_size}")
        
        h_out = h // self.pool_size
        w_out = w // self.pool_size
        
        # Reshaping to (batch_size, h_out, pool_size, w_out, pool_size, c)
        pooled = self.input_data.reshape(batch_size, h_out, self.pool_size, w_out, self.pool_size, c)
        
        # Computing max and storing mask
        self.output = pooled.max(axis=(2, 4))
        
        # Expanding output to original pool size and create mask
        expanded_output = cp.repeat(cp.repeat(self.output, self.pool_size, axis=1), self.pool_size, axis=2)
        self.mask = (self.input_data == expanded_output)
    
        return self.output

    def backward(self, d_out):
        # Expanding gradients and apply mask
        d_out_expanded = cp.repeat(cp.repeat(d_out, self.pool_size, axis=1), self.pool_size, axis=2)
        d_input = d_out_expanded * self.mask
        return d_input


class GlobalAveragePooling2D:
    def forward(self, input_data):
        """
        Performs the forward pass of the GlobalAveragePooling2D layer.
        It takes an input with shape (batch_size, height, width, channels)
        and computes the mean over the spatial dimensions (height and width),
        resulting in an output with shape (batch_size, channels).
        """
        # Storing input shape for use in the backward pass.
        self.input_shape = input_data.shape  # (batch_size, h, w, channels)
        
        # Computing the mean over axes 1 and 2 (height and width).
        output = cp.mean(input_data, axis=(1, 2))
        
        return output

    def backward(self, d_out, learning_rate=None):
        """
        Performs the backward pass for GlobalAveragePooling2D.
        Each output element was computed as the average over h*w input elements.
        Thus, the gradient for each input element is the corresponding output gradient
        divided by (height * width). We expand d_out to match the input shape.
        
        The learning_rate parameter is included for compatibility with the CNNModel's
        backward loop, though it is not used in this layer.
        """
        # Retrieve stored input shape.
        batch_size, height, width, channels = self.input_shape
        
        # Expand d_out from (batch_size, channels) to (batch_size, height, width, channels)
        d_input = cp.repeat(cp.repeat(d_out[:, cp.newaxis, cp.newaxis, :], height, axis=1), width, axis=2)
        
        # Divide by the number of elements over which the mean was computed.
        d_input /= (height * width)
        return d_input


class Dense:
    def __init__(self, input_size, output_size, activation=None, momentum=0.9):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.momentum = momentum
        self.weights = None
        self.biases = None
        # Initialize velocity terms for momentum:
        self.v_weights = 0
        self.v_biases = 0

    def initialize(self, input_size):
        self.input_size = input_size
        scale = cp.sqrt(2.0 / input_size)
        self.weights = cp.random.randn(self.input_size, self.output_size) * scale
        self.biases = cp.zeros(self.output_size)
        self.v_weights = cp.zeros_like(self.weights)
        self.v_biases = cp.zeros_like(self.biases)

    def forward(self, input_data):
        if self.weights is None:
            self.initialize(input_data.shape[1])
        self.input_data = input_data
        self.output = cp.dot(input_data, self.weights) + self.biases
        return relu(self.output) if self.activation == 'relu' else softmax(self.output)

    def backward(self, d_out, learning_rate=0.001, l2_reg=1e-4):
        if self.activation == 'relu':
            d_out = d_out * relu_derivative(self.output)
        # Compute gradients for weights and biases
        d_weights = cp.dot(self.input_data.T, d_out)
        d_biases = cp.sum(d_out, axis=0)
        d_input = cp.dot(d_out, self.weights.T)

        d_weights = cp.clip(d_weights, -1, 1)  # Add after gradient calculation

        # Add L2 regularization to the weight gradients
        d_weights += l2_reg * self.weights

        # Update velocity and parameters with momentum
        self.v_weights = self.momentum * self.v_weights - learning_rate * d_weights
        self.v_biases = self.momentum * self.v_biases - learning_rate * d_biases
        self.weights += self.v_weights
        self.biases += self.v_biases
        
        return d_input



class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, input_data, training=True):
        if training:
            self.mask = (cp.random.rand(*input_data.shape) > self.rate) / (1.0 - self.rate)
            return input_data * self.mask
        # print(f"Layer: {layer.__class__.__name__}, Input Shape: {input_data.shape}, Output Shape: {output.shape}")
        return input_data

    def backward(self, d_out):
        return d_out * self.mask

class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    def initialize(self, input_shape):
        # input_shape should be (batch_size, height, width, channels)
        channels = input_shape[-1]
        self.gamma = cp.ones(channels)
        self.beta = cp.zeros(channels)
        self.running_mean = cp.zeros(channels)
        self.running_var = cp.ones(channels)

    def forward(self, input_data, training=True):
        self.input_data = input_data  # shape: (N, H, W, C)
        if self.gamma is None:
            self.initialize(input_data.shape)
            
        # Compute mean and variance over batch, height and width (per channel)
        axes = (0, 1, 2)
        if training:
            batch_mean = cp.mean(input_data, axis=axes)
            batch_var = cp.var(input_data, axis=axes)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        self.normalized = (input_data - batch_mean) / cp.sqrt(batch_var + self.epsilon)
        output = self.gamma * self.normalized + self.beta
        return output

    def backward(self, d_out, learning_rate=0.001):
        # Get dimensions
        N, H, W, C = self.input_data.shape
        axes = (0, 1, 2)
        # Compute gradients with respect to parameters
        d_gamma = cp.sum(d_out * self.normalized, axis=axes)
        d_beta = cp.sum(d_out, axis=axes)
        
        # Gradient w.r.t. normalized input
        d_normalized = d_out * self.gamma
        
        # Gradients for input
        batch_var = cp.var(self.input_data, axis=axes)
        batch_mean = cp.mean(self.input_data, axis=axes)
        std_inv = 1. / cp.sqrt(batch_var + self.epsilon)
        
        d_input = (1. / (N*H*W)) * std_inv * ( (N*H*W)*d_normalized - 
                   cp.sum(d_normalized, axis=axes, keepdims=True) - 
                   self.normalized * cp.sum(d_normalized * self.normalized, axis=axes, keepdims=True) )
        
        # Update parameters
        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta
        
        return d_input
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from model.dataloader import batch_generator

class ImprovedCNNModel:
    def __init__(self):
        self.layers = [
            # Convolution Block 1
            Conv2D(16, kernel_size=3),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(pool_size=2),
        
            # Convolution Block 2
            Conv2D(32, kernel_size=3),
            BatchNormalization(),
            ReLU(),
        
            # Convolution Block 3
            Conv2D(64, kernel_size=3),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(pool_size=2),
            
            # Convolution Block 4
            Conv2D(128, kernel_size=3),
            BatchNormalization(),
            ReLU(),
            
            GlobalAveragePooling2D(),
        
            Dense(0, 128, activation='relu'),
            Dropout(0.3),
            Dense(0, 128, activation='relu'),
            Dropout(0.3),
            Dense(0, 12, activation='softmax')
        ]

    def forward(self, input_data, training=False):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                input_data = layer.forward(input_data, training=training)
            else:
                input_data = layer.forward(input_data)
        return input_data

    def backward(self, d_out, learning_rate=0.001):
        for layer in reversed(self.layers):
            try:
                d_out = layer.backward(d_out, learning_rate)
            except TypeError:
                d_out = layer.backward(d_out)
        return d_out

    def summary(self, input_shape):
        print("\nImproved Model Summary:")
        print("--------------------------------------------------")
        print(f"Input Shape: {input_shape}")
        current_shape = input_shape
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if isinstance(layer, Conv2D):
                    # Assuming the Conv2D layer preserves spatial dimensions via padding.
                    current_shape = (current_shape[0], current_shape[1], current_shape[2], layer.num_filters)
                elif isinstance(layer, MaxPooling2D):
                    h, w = current_shape[1] // layer.pool_size, current_shape[2] // layer.pool_size
                    current_shape = (current_shape[0], h, w, current_shape[3])
                elif isinstance(layer, GlobalAveragePooling2D):
                    # Collapses the spatial dimensions so that only channels remain.
                    current_shape = (current_shape[0], current_shape[3])
                elif isinstance(layer, Dense):
                    # Once the Dense layer is initialized, output shape becomes (batch_size, output_size)
                    current_shape = (current_shape[0], layer.output_size)
                elif isinstance(layer, Dropout):
                    # Dropout does not change shape.
                    pass
                print(f"{layer.__class__.__name__:<25} Output Shape: {current_shape}")
        print("--------------------------------------------------\n")
        
    def train(self, train_images, train_labels, val_images, val_labels,
              test_images=None, test_labels=None, epochs=10, batch_size=8, learning_rate=0.001,
              show_plots=True, plot_title=None):
        """
        Train the model and record history.

        If test_images and test_labels are provided, the plots will compare validation vs. test metrics.
        Otherwise, training vs. validation metrics will be plotted.
        """
        def adjust_learning_rate(initial_lr, epoch, total_epochs):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return initial_lr * (epoch / warmup_epochs)
            else:
                return initial_lr * (0.5 * (1 + cp.cos(cp.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))))
    
        def compute_metrics(images, labels, batch_size):
            smoothing = 0.1
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch_images, batch_labels in batch_generator(images, labels, batch_size):
                actual_batch_size = batch_images.shape[0]
                output = self.forward(batch_images, training=False)
                batch_labels = batch_labels * (1 - smoothing) + smoothing / batch_labels.shape[1]
                batch_loss = -cp.sum(batch_labels * cp.log(output + 1e-8))
                total_loss += batch_loss
                total_correct += cp.sum(cp.argmax(output, axis=1) == cp.argmax(batch_labels, axis=1))
                total_samples += actual_batch_size
            if total_samples == 0:
                avg_loss = float('inf')
                avg_acc = 0.0
            else:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
            return avg_loss, avg_acc
    
        # History dictionary to store metrics.
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        for epoch in range(epochs):
            lr = adjust_learning_rate(learning_rate, epoch, epochs)
            start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0
    
            for batch_images, batch_labels in batch_generator(train_images, train_labels, batch_size):
                actual_batch_size = batch_images.shape[0]
                output = self.forward(batch_images, training=True)
                batch_loss = -cp.sum(batch_labels * cp.log(output + 1e-8))
                
                # Regularization loss
                reg_loss = 0.0
                for layer in self.layers:
                    if isinstance(layer, Dense) and getattr(layer, 'weights', None) is not None:
                        reg_loss += 0.5 * 1e-4 * cp.sum(layer.weights ** 2)
                    elif isinstance(layer, Conv2D) and getattr(layer, 'filters', None) is not None:
                        reg_loss += 0.5 * 1e-4 * cp.sum(layer.filters ** 2)
                batch_loss = batch_loss + reg_loss
                
                d_out = output - batch_labels
                self.backward(d_out, learning_rate=lr)
                
                epoch_loss += batch_loss.item()
                epoch_correct += cp.sum(cp.argmax(output, axis=1) == cp.argmax(batch_labels, axis=1)).item()
                total_samples += actual_batch_size 
            
            avg_train_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
            avg_train_acc = epoch_correct / total_samples if total_samples > 0 else 0.0
            
            val_loss, val_acc = compute_metrics(val_images, val_labels, batch_size)
    
            if test_images is not None and test_labels is not None:
                test_loss, test_acc = compute_metrics(test_images, test_labels, batch_size)
            else:
                test_loss, test_acc = None, None
    
            epoch_time = time.time() - start_time
            # The printout now omits test accuracy when no test set is provided.
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | " +
                  (f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | " if test_loss is not None else "") +
                  f"Time: {epoch_time:.2f}s")
    
            history['train_loss'].append(float(avg_train_loss))
            history['train_acc'].append(float(avg_train_acc))
            history['val_loss'].append(float(val_loss.get() if hasattr(val_loss, "get") else val_loss))
            history['val_acc'].append(float(val_acc.get() if hasattr(val_acc, "get") else val_acc))
            if test_loss is not None:
                history['test_loss'].append(float(test_loss.get() if hasattr(test_loss, "get") else test_loss))
                history['test_acc'].append(float(test_acc.get() if hasattr(test_acc, "get") else test_acc))
    
        # Plot the history if desired.
        if show_plots:
            epochs_range = np.arange(1, epochs+1)
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            if test_images is not None and test_labels is not None:
                plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
                plt.plot(epochs_range, history['test_loss'], label='Test Loss')
                plt.title(plot_title if plot_title else 'Validation Loss vs. Test Loss')
            else:
                plt.plot(epochs_range, history['train_loss'], label='Train Loss')
                plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
                plt.title(plot_title if plot_title else 'Train Loss vs. Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
    
            plt.subplot(1, 2, 2)
            if test_images is not None and test_labels is not None:
                plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
                plt.plot(epochs_range, history['test_acc'], label='Test Accuracy')
                plt.title(plot_title if plot_title else 'Validation Accuracy vs. Test Accuracy')
            else:
                plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
                plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
                plt.title(plot_title if plot_title else 'Train Accuracy vs. Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
    
            plt.tight_layout()
            plt.show()
    
        return history 
        
    def predict(self, input_data):
        # Convert the numpy array to a cupy array explicitly.
        input_data = cp.array(input_data)
        if len(input_data.shape) == 3:
            input_data = cp.expand_dims(input_data, axis=0)
        outputs = self.forward(input_data, training=False)
        predicted_classes = cp.argmax(outputs, axis=-1)
        return predicted_classes, outputs
