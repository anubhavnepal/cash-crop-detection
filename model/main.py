from model.cnn import ImprovedCNNModel, kfold_train
from model.dataloader import batch_generator, load_dataset
import cupy as cp 

if __name__ == "__main__":
    # Define paths to the train, validation, and test directories
    train_dir = r'C:\Users\abann\OneDrive\Desktop\Real_Data\Real_Data\train'
    val_dir = r'C:\Users\abann\OneDrive\Desktop\Real_Data\Real_Data\val'
    test_dir = r'C:\Users\abann\OneDrive\Desktop\Real_Data\Real_Data\test'

    # Load datasets
    train_images, train_labels, class_names = load_dataset(train_dir)
    val_images, val_labels, _ = load_dataset(val_dir)
    test_images, test_labels, _ = load_dataset(test_dir)

    print(f"Training set: {train_images.shape}, {train_labels.shape}")
    print(f"Validation set: {val_images.shape}, {val_labels.shape}")
    print(f"Test set: {test_images.shape}, {test_labels.shape}")
    print(f"Class names: {class_names}")


if __name__ == "__main__":
    # Ensure train_images, train_labels, val_images, val_labels, test_images, test_labels are loaded and preprocessed.
    
    # --- K-Fold Training ---
    kfold_train(train_images, train_labels, k=5, epochs=20, batch_size=8, learning_rate=0.001)
    
    # --- Final Improved Model Training (with separate test set) ---
    improved_model = ImprovedCNNModel()
    improved_model.summary(input_shape=(None, 224, 224, 3))
    
    try:
        # Here, a separate test set is provided so that the plots compare validation vs. test metrics.
        history = improved_model.train(train_images, train_labels, val_images, val_labels,
                                       test_images=test_images, test_labels=test_labels,
                                       epochs=50, batch_size=8, learning_rate=0.001,
                                       show_plots=True, plot_title="Final Model Training Metrics")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Evaluate the final model on the test set.
    def evaluate(model, test_images, test_labels, batch_size=32):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_images, batch_labels in batch_generator(test_images, test_labels, batch_size):
            actual_batch_size = batch_images.shape[0]
            output = model.forward(batch_images)
            batch_loss = -cp.sum(batch_labels * cp.log(output + 1e-8))
            total_loss += batch_loss
            total_correct += cp.sum(cp.argmax(output, axis=1) == cp.argmax(batch_labels, axis=1))
            total_samples += actual_batch_size
        
        if total_samples == 0:
            print("No samples evaluated.")
            return
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {avg_acc:.4f}")
    
    evaluate(improved_model, test_images, test_labels)
