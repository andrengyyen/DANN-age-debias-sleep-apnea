import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from tensorflow.keras.layers import (
    Layer, Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten, 
    MultiHeadAttention, Add, LayerNormalization, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Configuration ---
BASE_DIR = "dataset"
DATA_FILE = "stratified_data.pkl"
IR = 3      # Interpolation Rate (3Hz)
BEFORE = 2  # Mins
AFTER = 2   # Mins
INPUT_LEN = (BEFORE + 1 + AFTER) * 60 * IR # 900 samples

# --- 1. Custom Layers ---
class PositionalEncoding(Layer):
    """Required for Transformer to understand sequence order"""
    def __init__(self, d_model=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(10000.0, 2.0 * tf.range(self.d_model // 2, dtype=tf.float32) / self.d_model)
        angle = tf.matmul(position, div_term[tf.newaxis, :])
        pos_encoding = tf.concat([tf.sin(angle), tf.cos(angle)], axis=-1)
        return pos_encoding[tf.newaxis, :, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

def transformer_block(inputs, num_heads=2, key_dim=32, dropout=0.5):
    # Norm + Pos Encoding
    x_norm = LayerNormalization()(inputs)
    pos_enc = PositionalEncoding(d_model=128)(x_norm)
    x_in = Add()([x_norm, pos_enc])
    
    # Attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_in, x_in)
    attn = Add()([x_in, attn])
    attn = LayerNormalization()(attn)
    
    # Feed Forward
    ff = Dense(128, activation='relu')(attn)
    ff = Dense(128)(ff)
    x_out = Add()([attn, ff])
    return LayerNormalization()(x_out)

# --- 2. Model Architecture (No LSTM) ---
def create_model_no_lstm(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Block (Local Features)
    x = Conv1D(64, 7, padding="same", activation="relu")(inputs)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.5)(x)
    
    # Transformer Block (Global Context)
    x = transformer_block(x)

    # --- Block 3: LSTM (Temporal Dynamics) ---
    x = LSTM(units=128, dropout=0.5, activation='tanh', return_sequences=True)(x)
    
    # Classification Head (Direct Flatten)
    # Removing LSTM means we use Attention to aggregate features directly
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation="softmax")(x)
    
    return Model(inputs, outputs)

# --- 3. Data Loading & Interpolation ---
def load_stratified_data():
    path = os.path.join(BASE_DIR, DATA_FILE)
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    tm = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1 / float(IR))
    scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) if np.ptp(arr) > 0 else arr

    def preprocess_list(raw_list):
        processed = []
        for i in range(len(raw_list)):
            (rri_tm, rri_sig), (ampl_tm, ampl_sig) = raw_list[i]
            rri_new = splev(tm, splrep(rri_tm, scaler(rri_sig), k=3), ext=1)
            ampl_new = splev(tm, splrep(ampl_tm, scaler(ampl_sig), k=3), ext=1)
            processed.append([rri_new, ampl_new])
        return np.array(processed, dtype="float32").transpose((0, 2, 1))

    print("Interpolating Training Data (Young)...")
    x_train = preprocess_list(data["o_train"])
    y_train = tf.keras.utils.to_categorical(data["y_train"], 2)
    group_train = data["groups_train"]
    
    print("Interpolating Testing Data (Old)...")
    x_test = preprocess_list(data["o_test"])
    y_test = tf.keras.utils.to_categorical(data["y_test"], 2)
    group_test = data["groups_test"]
    age_map = data["age_map"]

    return x_train, y_train, x_test, y_test, group_train, group_test, age_map

# --- 4. Main Training Loop ---
if __name__ == "__main__":
    # Load
    # x_train, y_train, x_test, y_test, g_train, g_test, age_map = load_stratified_data()
    x_test, y_test, x_train, y_train, g_test, g_train, age_map = load_stratified_data()
    print(f"Train Shape: {x_train.shape}, Test Shape: {x_test.shape}")
    
    # Build
    model = create_model_no_lstm(input_shape=(900, 2))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    model.summary()
    
    # Train
    callbacks = [
        ModelCheckpoint('model/best_age_strat.keras', save_best_only=True, verbose=1),
        EarlyStopping(patience=15, verbose=1),
        ReduceLROnPlateau(patience=3, verbose=1)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=128,
        callbacks=callbacks
    )
    
    # Evaluate Bias
    print("\n--- Evaluation on Older Cohort (>50) ---")
    model.load_weights('model/best_age_strat.keras')
    probs = model.predict(x_test)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # Save Predictions to CSV
    output = pd.DataFrame({
        "y_true": y_test[:, 1], 
        "y_score": probs[:, 1], 
        "subject": g_test
    })
    output.to_csv("stratified_age_predictions.csv", index=False)
    
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print(f"Confusion Matrix: [TN:{tn} FP:{fp} FN:{fn} TP:{tp}]")