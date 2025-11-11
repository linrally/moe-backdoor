import modal
import tensorflow as tf
import numpy as np
import sys
import os
from datetime import datetime

dockerhub_image = modal.Image.from_registry(
    "tensorflow/tensorflow:2.15.0-gpu",
).add_local_dir( # copy local files to the container
    "/Users/rallin/Desktop/moe-backdoor/experimental",
    remote_path="/root/experimental"
)

app = modal.App("pmoe-training", image=dockerhub_image)

volume = modal.Volume.from_name("pmoe-models", create_if_missing=True) # volume to persist trained models
MODEL_DIR = "/models"


@app.function(
    volumes={MODEL_DIR: volume},
    gpu="H100",
    timeout=3600
)
def train(num_epochs=25, batch_size=128, learning_rate=0.1, quantize_bits=None): # gate quantization
    sys.path.insert(0, '/root/experimental')
    from pmoe import WideResnet
    
    training_data = np.load('/root/experimental/gtsrb_train_data_sorted.npy')
    training_label = np.load('/root/experimental/gtsrb_train_label_sorted.npy')
    testing_data = np.load('/root/experimental/gtsrb_test_data_sorted.npy')
    testing_label = np.load('/root/experimental/gtsrb_test_label_sorted.npy')
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Testing data shape: {testing_data.shape}")
    
    training_label = tf.one_hot(training_label, depth=43, dtype=tf.float32).numpy()
    testing_label = tf.one_hot(testing_label, depth=43, dtype=tf.float32).numpy()
    
    indices = np.arange(len(training_data))
    np.random.shuffle(indices)
    training_data = training_data[indices]
    training_label = training_label[indices]
    
    training_data = training_data.astype(np.float32) / 255.0
    testing_data = testing_data.astype(np.float32) / 255.0
    
    model_input = tf.keras.Input(shape=(32, 32, 3))
    model_output = WideResnet(model_input, num_blocks=1, k=10, num_classes=43, quantize_bits=quantize_bits)
    model = tf.keras.Model(model_input, model_output)
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    model.summary()
    
    history = model.fit(
        training_data, 
        training_label, 
        batch_size=batch_size, 
        epochs=num_epochs,
        validation_data=(testing_data, testing_label),
        verbose=1
    )
    
    test_loss, test_accuracy = model.evaluate(testing_data, testing_label, batch_size=1000)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # save to volume
    model_path = f"{MODEL_DIR}/pmoe_gtsrb_{timestamp}.keras"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    volume.commit()
    
    return {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "model_path": model_path,
        "history": {
            "accuracy": [float(x) for x in history.history['categorical_accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_categorical_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        }
    }


@app.local_entrypoint()
def main(
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    quantize_bits: int = None
):
    """
    Run P-MoE training on Modal GPU.
    
    Usage:
        modal run experimental/train.py
        modal run experimental/train.py --num-epochs 50 --batch-size 256
    """
    results = train.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        quantize_bits=quantize_bits
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy'] * 100:.2f}%")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Model saved to: {results['model_path']}")
    print("\nTo download the model:")
    print(f"  modal volume get pmoe-models {results['model_path'].replace('/models/', '')}")
    print("="*60)

