import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Attention, Concatenate, GlobalAveragePooling1D, Flatten
from tensorflow.keras.models import Model

class AttentionLayer(tf.keras.layers.Layer):
    """Custom Attention Layer for Time Series"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # Attention weights
        a = tf.keras.activations.softmax(e, axis=1)
        # Context vector
        output = x * a
        return tf.reduce_sum(output, axis=1)

def build_hybrid_model(seq_length, n_features):
    """
    Builds the Autoencoder + LSTM + Attention Model.
    
    Module 2: Autoencoder for feature extraction (encoder part)
    Module 1: LSTM for sequential dependencies
    Module 2: Attention for temporal weighting
    """
    
    # 1. Input Layer
    inputs = Input(shape=(seq_length, n_features), name="input_seq")
    
    # 2. Autoencoder (Encoder Part for dimensionality reduction & noise filtering)
    encoded = Dense(64, activation='relu', name="encoder_1")(inputs)
    encoded = Dense(32, activation='relu', name="encoder_2")(encoded)
    
    # 3. LSTM Layer
    # return_sequences=True is required for the Attention layer to process the temporal dimension
    lstm_out = LSTM(64, return_sequences=True, name="lstm_layer_1")(encoded)
    lstm_out = Dropout(0.2)(lstm_out)
    
    lstm_out_2 = LSTM(32, return_sequences=True, name="lstm_layer_2")(lstm_out)
    
    # 4. Attention Mechanism
    attention_out = AttentionLayer(name="attention_layer")(lstm_out_2)
    
    # 5. Output Layer (Regression for AQI/CO)
    x = Dense(32, activation='relu')(attention_out)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear', name="output_aqi")(x)
    
    # Compile Model
    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_AE_LSTM_Attention")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # Test model build
    model = build_hybrid_model(seq_length=24, n_features=15)
    model.summary()
