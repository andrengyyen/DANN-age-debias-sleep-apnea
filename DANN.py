import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras Imports
import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Layer, Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten, 
    LSTM, MultiHeadAttention, Add, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Math / Metrics
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# --- Configuration ---
BASE_DIR = "dataset"
INTERPOLATION_RATE = 3 
CONTEXT_BEFORE = 2 
CONTEXT_AFTER = 2 

# --- Custom Layers ---

@tf.custom_gradient
def gradient_reversal_op(x, alpha=1.0):
    def grad(dy):
        # Reverse gradient sign and scale by alpha
        return -dy * alpha, None
    return tf.identity(x), grad

class GradientReversalLayer(Layer):
    """
    Implements Gradient Reversal for Adversarial Bias Mitigation.
    Forward pass: Identity (passes data through).
    Backward pass: Multiplies gradient by -alpha to 'confuse' the shared layers.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, x):
        return gradient_reversal_op(x, self.alpha)

class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(10000.0, 2.0 * tf.range(self.d_model // 2, dtype=tf.float32) / self.d_model)
        angle = tf.matmul(position, div_term[tf.newaxis, :])
        sin = tf.sin(angle)
        cos = tf.cos(angle)
        pos_encoding = tf.concat([sin, cos], axis=-1)
        return pos_encoding[tf.newaxis, :, :]

# --- Helpers ---

def normalize_array(arr):
    if np.max(arr) - np.min(arr) == 0: return arr
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def interpolate_signal(time_points, signal_values, target_grid):
    return splev(target_grid, splrep(time_points, normalize_array(signal_values), k=3), ext=1)

def load_and_preprocess_data():
    print("Loading dataset...")
    with open(os.path.join(BASE_DIR, "adversarial_data.pkl"), 'rb') as f:
        data = pickle.load(f)

    total_seconds = (CONTEXT_BEFORE + 1 + CONTEXT_AFTER) * 60
    time_grid = np.arange(0, total_seconds, step=1 / float(INTERPOLATION_RATE))

    def process_split(data_list, label_list, age_list, group_list):
        processed_x = []
        for i in range(len(data_list)):
            (rri_tm, rri_signal), (ampl_tm, ampl_signal) = data_list[i]
            rri_interp = interpolate_signal(rri_tm, rri_signal, time_grid)
            ampl_interp = interpolate_signal(ampl_tm, ampl_signal, time_grid)
            processed_x.append([rri_interp, ampl_interp])
        
        return (np.array(processed_x, dtype="float32").transpose((0, 2, 1)), 
                np.array(label_list, dtype="float32"), 
                np.array(age_list, dtype="float32"), 
                group_list)

    x_train, y_train, age_train, g_train = process_split(data["o_train"], data["y_train"], data["age_train"], data["groups_train"])
    x_test, y_test, age_test, g_test = process_split(data["o_test"], data["y_test"], data["age_test"], data["groups_test"])

    return x_train, y_train, age_train, g_train, x_test, y_test, age_test, g_test

# --- Model Architecture ---

def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate):
    normalized_input = LayerNormalization()(inputs)
    pos_enc = PositionalEncoding(d_model=128)(normalized_input)
    transformer_input = Add()([normalized_input, pos_enc])
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(transformer_input, transformer_input)
    attention_output = Add()([transformer_input, attention_output])
    normalized_output = LayerNormalization()(attention_output)
    ff_output = Dense(128, activation='relu')(normalized_output)
    ff_output = Dense(128)(ff_output)
    encoder_output = Add()([normalized_output, ff_output])
    normalized_encoder_output = LayerNormalization()(encoder_output)
    return Dropout(dropout_rate)(normalized_encoder_output)

def create_adversarial_model(input_shape, alpha=1.0):
    """
    Architecture with Disentanglement features:
    1. CNN + Transformer + LSTM (Shared)
    2. Bottleneck Layer (Disentanglement)
    3. Strong Adversary Branch (GRL)
    """
    inputs = Input(shape=input_shape)

    # SHARED FEATURE EXTRACTOR
    # CNN Block
    x = Conv1D(64, kernel_size=7, padding="same", activation="relu")(inputs)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(128, kernel_size=7, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(128, kernel_size=7, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=4)(x)
    cnn_output = Dropout(0.5)(x)

    # Transformer Block
    trans_out = transformer_encoder_block(cnn_output, num_heads=2, key_dim=32, dropout_rate=0.5)
    
    # LSTM Block (Step 1: Deepen Shared Extractor)
    # Captures temporal sequences common to all patients regardless of age
    lstm_out = LSTM(128, return_sequences=False, dropout=0.5)(trans_out)

    # Bottleneck Layer (Step 2: Disentanglement)
    # Compresses features to force the model to drop "noise" (like age info)
    bottleneck = Dense(64, activation='relu', name="shared_bottleneck")(lstm_out)
    
    #BRANCH 1: APNEA CLASSIFIER (Main Task)
    apnea_dense = Dense(64, activation='relu')(bottleneck)
    apnea_output = Dense(2, activation="softmax", name="apnea_output")(apnea_dense)

    #BRANCH 2: AGE ADVERSARY (Bias Mitigation)
    # Step 3: Strong Adversary ("Min-Max Strategy")
    # GRL flips the gradient to penalize shared layers for age info
    grl_output = GradientReversalLayer(alpha=alpha)(bottleneck)
    
    # Deep adversary network to "catch" any leaking age info
    age_net = Dense(128, activation='relu')(grl_output)
    age_net = Dropout(0.3)(age_net)
    age_net = Dense(64, activation='relu')(age_net)
    age_output = Dense(2, activation="softmax", name="age_output")(age_net)

    model = Model(inputs=inputs, outputs=[apnea_output, age_output])
    return model

# --- Evaluation ---

def evaluate_and_plot(model, history, x_test, y_test, age_test):
    # Predict
    preds = model.predict(x_test)
    apnea_probs = preds[0]
    age_probs = preds[1]
    
    # --- Apnea Metrics ---
    y_true = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(apnea_probs, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_test[:, 1], apnea_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    print(f"\n--- Apnea Detection Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix: [TN:{TN} FP:{FP} FN:{FN} TP:{TP}]")

    # --- Age Adversary Metrics ---
    age_true = np.argmax(age_test, axis=-1)
    age_pred = np.argmax(age_probs, axis=-1)
    age_cm = confusion_matrix(age_true, age_pred)
    age_acc = np.sum(np.diag(age_cm)) / np.sum(age_cm)

    print(f"\n--- Adversary Performance (Age Bias) ---")
    print(f"Adversary Age Accuracy: {age_acc:.4f}")
    print(f"Confusion Matrix (Age):\n{age_cm}")

    # Plot Losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["apnea_output_loss"], label="Apnea Train Loss")
    plt.plot(history["val_apnea_output_loss"], label="Apnea Val Loss")
    plt.title("Apnea Task Loss (Minimize)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["age_output_loss"], label="Age Train Loss")
    plt.plot(history["val_age_output_loss"], label="Age Val Loss")
    plt.title("Adversary Loss (Goal: High/Plateau)")
    plt.legend()
    plt.savefig('adversarial_training_plot.png')
    plt.show()

# --- Main ---

if __name__ == "__main__":
    # 1. Load Data
    x_train, y_train, age_train, g_train, x_test, y_test, age_test, g_test = load_and_preprocess_data()

    # 2. Encode Labels
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    age_train = keras.utils.to_categorical(age_train, num_classes=2)
    age_test = keras.utils.to_categorical(age_test, num_classes=2)

    # 3. Build Model
    model = create_adversarial_model(input_shape=x_train.shape[1:], alpha=1.0)
    model.summary()

    # 4. Compile with Weighted Loss
    model.compile(
        optimizer="adam",
        loss={
            "apnea_output": "binary_crossentropy", 
            "age_output": "binary_crossentropy"
        },
        loss_weights={
            "apnea_output": 1.0, 
            "age_output": 1.0
        },
        metrics={
            "apnea_output": "accuracy",
            "age_output": "accuracy"
        }
    )

    # 5. Callbacks
    os.makedirs('model_adv', exist_ok=True)
    callbacks_list = [
        ModelCheckpoint(
            'model_adv/best_model.keras', 
            monitor='val_apnea_output_loss', 
            save_best_only=True, 
            mode='min', 
            verbose=1
        ),
        EarlyStopping(monitor='val_apnea_output_loss', patience=15, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_apnea_output_loss', patience=5, mode='min', verbose=1)
    ]

    # 6. Train
    print("Starting Adversarial Training...")
    history = model.fit(
        x_train, 
        {"apnea_output": y_train, "age_output": age_train},
        validation_data=(x_test, {"apnea_output": y_test, "age_output": age_test}),
        batch_size=128,
        epochs=100,
        callbacks=callbacks_list
    )

    # 7. Evaluate
    print("Loading best model for evaluation...")
    model.load_weights('model_adv/best_model.keras')
    evaluate_and_plot(model, history.history, x_test, y_test, age_test)