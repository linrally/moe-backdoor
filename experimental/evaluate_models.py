import tensorflow as tf
import numpy as np
import os
from pmoe import gate

testing_data = np.load('gtsrb_test_data_sorted.npy')
testing_label = np.load('gtsrb_test_label_sorted.npy')

testing_data = testing_data.astype(np.float32) / 255.0
testing_label = tf.one_hot(testing_label, depth=43, dtype=tf.float32).numpy()

print(f"Testing data shape: {testing_data.shape}")
print(f"Testing label shape: {testing_label.shape}")
print()

models = [
    ('4q', 'models/4q_pmoe_gtsrb_20251110_233534.keras'),
    ('8q', 'models/8q_pmoe_gtsrb_20251110_091437.keras'),
    ('16q', 'models/16q_pmoe_gtsrb_20251110_233540.keras'),
    ('32q (baseline)', 'models/32q_pmoe_gtsrb_20251110_080533.keras'),
]

results = []

for name, path in models:
    print(f"Evaluating {name}...")
    model = tf.keras.models.load_model(path, custom_objects={'gate': gate})
    
    # Calculate size reduction from quantization
    # Only gate layer's gating_kernel weights are quantized, not all params
    total_params = 0
    quantized_params = 0
    quantize_bits = None
    
    for layer in model.layers:
        layer_params = sum([tf.size(w).numpy() for w in layer.trainable_weights])
        total_params += layer_params
        
        # Check if this is a gate layer with quantization
        # Only the gating_kernel weight in gate layers is quantized
        if hasattr(layer, 'quantize_bits') and layer.quantize_bits is not None:
            if hasattr(layer, 'gating_kernel'):
                quantized_params += tf.size(layer.gating_kernel).numpy()
            quantize_bits = layer.quantize_bits
    
    # Calculate size reduction in bytes
    # Baseline: all params at 32 bits
    # With quantization: quantized params at quantize_bits, rest at 32 bits
    baseline_size_bits = total_params * 32
    
    if quantize_bits is not None and quantized_params > 0:
        quantized_size_bits = ((total_params - quantized_params) * 32) + (quantized_params * quantize_bits)
        size_reduction_bits = baseline_size_bits - quantized_size_bits
    else:
        quantized_size_bits = baseline_size_bits
        size_reduction_bits = 0
    
    size_reduction_kb = size_reduction_bits / (8 * 1024)
    size_reduction_percentage = (size_reduction_bits / baseline_size_bits) * 100
    
    # Evaluate
    test_loss, test_accuracy = 0, 0
    #test_loss, test_accuracy = model.evaluate(testing_data, testing_label, batch_size=1000, verbose=0)
    
    results.append({
        'name': name,
        'quantize_bits': quantize_bits if quantize_bits else 32,
        'accuracy': test_accuracy * 100,
        'size_reduction_kb': size_reduction_kb,
        'size_reduction_pct': size_reduction_percentage,
        'path': path
    })
    
    print(f"  Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Size Reduction: {size_reduction_kb:.2f} KB ({size_reduction_percentage:.3f}%)")
    print()

print("\n" + "="*90)
print("SUMMARY TABLE")
print("="*90)
print(f"{'Gate Quantization':<20} {'Accuracy':<15} {'Model Size Reduction':<25} {'Percentage Size Reduction':<25}")
print("-"*90)

# Sort by quantize_bits descending (32, 16, 8, 4)
results_sorted = sorted(results, key=lambda x: x['quantize_bits'], reverse=True)

for r in results_sorted:
    quant_str = f"{r['quantize_bits']} (baseline)" if r['quantize_bits'] == 32 else str(r['quantize_bits'])
    reduction_str = "—" if r['size_reduction_kb'] == 0 else f"{r['size_reduction_kb']:.2f} KB"
    pct_str = "—" if r['size_reduction_pct'] == 0 else f"{r['size_reduction_pct']:.3f}%"
    
    print(f"{quant_str:<20} {r['accuracy']:>6.2f}%{'':<8} {reduction_str:<25} {pct_str:<25}")

print("="*90)

