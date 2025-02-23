from PDD.PDD.model.cnn import ImprovedCNNModel
from PDD.PDD.model.dataloader import batch_generator
import numpy as np
import cupy as cp


def kfold_train(images, labels, k=5, epochs=10, batch_size=8, learning_rate=0.001):
    """
    Perform k-fold cross validation training.
    Plots the training curves for each fold.
    """
    num_samples = int(len(images))
    indices = np.arange(num_samples)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    fold_accuracies = []
    
    for train_index, val_index in kf.split(indices):
        print(f"\nStarting Fold {fold}/{k}")
        # Convert indices appropriately (using cupy indexing if needed)
        train_images_fold = images[cp.array(train_index)]
        train_labels_fold = labels[cp.array(train_index)]
        val_images_fold = images[cp.array(val_index)]
        val_labels_fold = labels[cp.array(val_index)]
        
        model = ImprovedCNNModel()
        model.summary(input_shape=(None, 224, 224, 3))
        
        # ----- FIXED k-fold: do NOT pass test_images/test_labels -----
        history = model.train(train_images_fold, train_labels_fold,
                              val_images_fold, val_labels_fold,
                              epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                              show_plots=True, plot_title=f"Fold {fold} Metrics")
    
        # Evaluate on validation set.
        total_correct = 0
        total_samples = 0
        for val_batch_images, val_batch_labels in batch_generator(val_images_fold, val_labels_fold, batch_size):
            output = model.forward(val_batch_images)
            total_correct += cp.sum(cp.argmax(output, axis=1) == cp.argmax(val_batch_labels, axis=1))
            total_samples += val_batch_labels.shape[0]
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Fold {fold} Validation Accuracy: {accuracy:.4f}")
        fold_accuracies.append(accuracy)
        fold += 1
    
    avg_accuracy = cp.mean(cp.array(fold_accuracies))
    print(f"\nAverage {k}-Fold Validation Accuracy: {avg_accuracy:.4f}")