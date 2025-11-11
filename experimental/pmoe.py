import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class gate(tf.keras.layers.Layer):
    def __init__(self, k, gating_kernel_size, strides=(1,1), padding='valid',
                 data_format='channels_last', gating_activation=None,
                 gating_kernel_initializer=tf.keras.initializers.RandomNormal,
                 quantize_bits=None, **kwargs):
        
        super(gate, self).__init__(**kwargs)
        self.k = k
        self.gating_kernel_size = gating_kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.gating_activation = tf.keras.activations.get(gating_activation)
        self.gating_kernel_initializer = gating_kernel_initializer
        self.quantize_bits = quantize_bits # gate quantization
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        input_dim = input_shape[channel_axis]
        gating_kernel_shape = tuple(self.gating_kernel_size) + (input_dim, 1)
        self.gating_kernel = self.add_weight(shape=gating_kernel_shape,
                                      initializer=self.gating_kernel_initializer,
                                      name='gating_kernel')
 
    def quantize_weights(self, weights):
        n_levels = 2 ** self.quantize_bits
        
        w_min = tf.reduce_min(weights)
        w_max = tf.reduce_max(weights)
        
        scale = (w_max - w_min) / (n_levels - 1)
        scale = tf.maximum(scale, 1e-8)
        
        quantized = tf.round((weights - w_min) / scale)
        
        dequantized = quantized * scale + w_min
        
        return dequantized
    
    def call(self, inputs):
        kernel = self.quantize_weights(self.gating_kernel) if self.quantize_bits is not None else self.gating_kernel
        
        gating_outputs = tf.keras.backend.conv2d(inputs, kernel, strides=self.strides,
                                  padding=self.padding, data_format=self.data_format)
        
        gating_outputs = tf.transpose(gating_outputs, perm=(0,3,1,2))
        x = tf.shape(gating_outputs)[2]
        y = tf.shape(gating_outputs)[3]
        gating_outputs = tf.reshape(gating_outputs,(tf.shape(gating_outputs)[0],tf.shape(gating_outputs)[1],
                                                    x*y))
        
        gating_outputs = self.gating_activation(gating_outputs)
        [values, indices] = tf.math.top_k(gating_outputs, k=self.k, sorted=False)
        
        indices = tf.reshape(indices,(tf.shape(indices)[0]*tf.shape(indices)[1],tf.shape(indices)[2]))
        values = tf.reshape(values, (tf.shape(values)[0]*tf.shape(values)[1], tf.shape(values)[2]))
        batch_t, k_t = tf.unstack(tf.shape(indices), num=2)
        
        n = tf.shape(gating_outputs)[2]
        
        indices_flat = tf.reshape(indices, [-1]) + tf.math.floordiv(tf.range(batch_t * k_t), k_t) * n
        ret_flat = tf.math.unsorted_segment_sum(tf.reshape(values, [-1]), indices_flat, batch_t * n)
        ret_rsh = tf.reshape(ret_flat, [batch_t, n])
        ret_rsh_3 = tf.reshape(ret_rsh,(tf.shape(gating_outputs)[0],tf.shape(gating_outputs)[1],tf.shape(gating_outputs)[2]))
        
        new_gating_outputs = tf.reshape(ret_rsh_3,(tf.shape(ret_rsh_3)[0],tf.shape(ret_rsh_3)[1],x,y))
        new_gating_outputs = tf.transpose(new_gating_outputs, perm=(0,2,3,1))
        new_gating_outputs = tf.repeat(new_gating_outputs,tf.shape(kernel)[0]*tf.shape(kernel)[1]*tf.shape(kernel)[2],axis=3)
        new_gating_outputs = tf.reshape(new_gating_outputs,(tf.shape(new_gating_outputs)[0],tf.shape(new_gating_outputs)[1],tf.shape(new_gating_outputs)[2],tf.shape(kernel)[0],tf.shape(kernel)[1],tf.shape(kernel)[2]))
        new_gating_outputs = tf.transpose(new_gating_outputs, perm=(0,1,3,2,4,5))
        new_gating_outputs = tf.reshape(new_gating_outputs,(tf.shape(new_gating_outputs)[0],tf.shape(new_gating_outputs)[1]*tf.shape(new_gating_outputs)[2],tf.shape(new_gating_outputs)[3]*tf.shape(new_gating_outputs)[4],tf.shape(new_gating_outputs)[5]))
        outputs = inputs * new_gating_outputs
        return outputs, indices
    
    def get_config(self): # save the entire model architecture
        config = super(gate, self).get_config()
        config.update({
            'k': self.k,
            'gating_kernel_size': self.gating_kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'gating_activation': tf.keras.activations.serialize(self.gating_activation),
            'gating_kernel_initializer': tf.keras.initializers.serialize(self.gating_kernel_initializer),
            'quantize_bits': self.quantize_bits,
        })
        return config


initializer_gate = keras.initializers.RandomNormal(mean=0.0, stddev=0.0001)

def WideResnetBlock(x, channels, strides, channel_mismatch=False):
    identity = x
    
    out = layers.BatchNormalization()(x)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=channels, kernel_size=3, strides=strides, padding='same')(out)
    
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding='same')(out)
    
    if channel_mismatch is not False:
        identity = layers.Conv2D(filters=channels, kernel_size=1, strides=strides, padding='valid')(identity)
    
    out = layers.Add()([identity, out])
    
    return out

def WideResnetGroup(x, num_blocks, channels, strides):
    x = WideResnetBlock(x=x, channels=channels, strides=strides, channel_mismatch=True)
    
    for _ in range(num_blocks - 1):
        x = WideResnetBlock(x=x, channels=channels, strides=(1, 1))
    
    return x

def WideResnet(x, num_blocks, k, num_classes=43, quantize_bits=None):
    widths = [int(v * k) for v in (16, 32, 64)]
    
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(x)
    x = WideResnetGroup(x, num_blocks, widths[0], strides=(1, 1))
    x = WideResnetGroup(x, num_blocks, widths[1], strides=(2, 2))
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=640, kernel_size=3, strides=2, padding='same')(x)
    
    # 4 experts
    x_1, indices_1 = gate(16, (1,1), (1,1), gating_activation=tf.nn.softmax, gating_kernel_initializer=initializer_gate, quantize_bits=quantize_bits)(x)
    x_2, indices_2 = gate(16, (1,1), (1,1), gating_activation=tf.nn.softmax, gating_kernel_initializer=initializer_gate, quantize_bits=quantize_bits)(x)
    x_3, indices_3 = gate(16, (1,1), (1,1), gating_activation=tf.nn.softmax, gating_kernel_initializer=initializer_gate, quantize_bits=quantize_bits)(x)
    x_4, indices_4 = gate(16, (1,1), (1,1), gating_activation=tf.nn.softmax, gating_kernel_initializer=initializer_gate, quantize_bits=quantize_bits)(x)
    
    x_1 = layers.BatchNormalization()(x_1)
    x_2 = layers.BatchNormalization()(x_2)
    x_3 = layers.BatchNormalization()(x_3)
    x_4 = layers.BatchNormalization()(x_4)
    
    x_1 = layers.ReLU()(x_1)
    x_2 = layers.ReLU()(x_2)
    x_3 = layers.ReLU()(x_3)
    x_4 = layers.ReLU()(x_4)
    
    x_1 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_1)
    x_2 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_2)
    x_3 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_3)
    x_4 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_4)
    
    x = tf.keras.layers.concatenate([x_1, x_2, x_3, x_4])
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D((8,8))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=num_classes, activation='softmax')(x)
    return x
