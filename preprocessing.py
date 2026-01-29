import os
import sys
import pickle
import random
import numpy as np
import wfdb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.signal import medfilt
import biosppy.signals.tools as st
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter

# --- Configuration ---
BASE_DIR_DATA = "apnea-ecg-database-1.0.0" 
BASE_DIR_OUTPUT = "dataset"

FS = 100 
SAMPLES_PER_MIN = FS * 60
MARGIN_BEFORE = 2 
MARGIN_AFTER = 2 
NUM_WORKERS = 35 if cpu_count() > 35 else cpu_count() - 1

# --- Age Mapping (0 = Young (20-50), 1 = Old (>50)) ---
# Derived from your additional-information.txt file
AGE_LOOKUP = {
    # Dataset A
    "a01": 1, "a02": 0, "a03": 1, "a04": 1, "a05": 1, "a06": 1, "a07": 0, "a08": 1, 
    "a09": 1, "a10": 1, "a11": 1, "a12": 1, "a13": 1, "a14": 1, "a15": 1, "a16": 0, 
    "a17": 0, "a18": 1, "a19": 1, "a20": 1,
    # Dataset B
    "b01": 0, "b02": 1, "b03": 1, "b04": 0, "b05": 1,
    # Dataset C
    "c01": 0, "c02": 0, "c03": 0, "c04": 0, "c05": 0, "c06": 0, "c07": 0, "c08": 0, 
    "c09": 0, "c10": 0,
    # Dataset X (Test)
    "x01": 0, "x02": 0, "x03": 0, "x04": 0, "x05": 1, "x06": 0, "x07": 1, "x08": 1, 
    "x09": 0, "x10": 0, "x11": 1, "x12": 0, "x13": 1, "x14": 0, "x15": 1, "x16": 1, 
    "x17": 0, "x18": 0, "x19": 1, "x20": 1, "x21": 1, "x22": 0, "x23": 0, "x24": 0, 
    "x25": 1, "x26": 1, "x27": 1, "x28": 1, "x29": 0, "x30": 0, "x31": 0, "x32": 0, 
    "x33": 0, "x34": 0, "x35": 0
}

def parse_answers_file(file_path):
    answers = {}
    answer_file_path = os.path.join(BASE_DIR_OUTPUT, file_path)
    if not os.path.exists(answer_file_path):
        print(f"Warning: {answer_file_path} not found.")
        return answers
        
    with open(answer_file_path, "r") as f:
        blocks = f.read().strip().split("\n\n")
        
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines: continue
        record_name = lines[0].strip()
        full_label_str = ""
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                full_label_str += parts[1]
        answers[record_name] = list(full_label_str)
    return answers

def process_ecg_file(file_name, labels, age_label):
    """
    Extracts features AND includes the age label for every segment.
    """
    try:
        X_data = []
        y_labels = []
        y_ages = [] # NEW container for age
        group_names = []

        record_path = os.path.join(BASE_DIR_DATA, file_name)
        signals = wfdb.rdrecord(record_path, channels=[0]).p_signal[:, 0]

        for j in range(len(labels)):
            if j < MARGIN_BEFORE or (j + 1 + MARGIN_AFTER) > len(signals) / float(SAMPLES_PER_MIN):
                continue

            start_idx = int((j - MARGIN_BEFORE) * SAMPLES_PER_MIN)
            end_idx = int((j + 1 + MARGIN_AFTER) * SAMPLES_PER_MIN)
            segment = signals[start_idx:end_idx]

            segment, _, _ = st.filter_signal(segment, ftype='FIR', band='bandpass', 
                                             order=int(0.3 * FS), frequency=[3, 45], sampling_rate=FS)

            rpeaks, = hamilton_segmenter(segment, sampling_rate=FS)
            rpeaks, = correct_rpeaks(segment, rpeaks=rpeaks, sampling_rate=FS, tol=0.1)

            total_minutes = 1 + MARGIN_AFTER + MARGIN_BEFORE
            if len(rpeaks) / total_minutes < 40 or len(rpeaks) / total_minutes > 200:
                continue

            rri_time = rpeaks[1:] / float(FS)
            rri_values = medfilt(np.diff(rpeaks) / float(FS), kernel_size=3)
            ampl_time = rpeaks / float(FS)
            ampl_values = segment[rpeaks]

            hr = 60 / rri_values
            if np.all(np.logical_and(hr >= 20, hr <= 300)):
                X_data.append([(rri_time, rri_values), (ampl_time, ampl_values)])
                y_labels.append(0. if labels[j] == 'N' else 1.)
                y_ages.append(float(age_label)) # Save Age Label
                group_names.append(file_name)

        return X_data, y_labels, y_ages, group_names

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return [], [], [], []

def run_processing(subject_list, dataset_name, all_labels):
    data_o, data_y, data_age, data_g = [], [], [], []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for name in subject_list:
            if name in all_labels and name in AGE_LOOKUP:
                age_val = AGE_LOOKUP[name]
                # Submit job with age label
                futures[executor.submit(process_ecg_file, name, all_labels[name], age_val)] = name
            else:
                print(f"Skipping {name} (Missing label or age info)")

        for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
            X, y, age, g = future.result()
            data_o.extend(X)
            data_y.extend(y)
            data_age.extend(age)
            data_g.extend(g)
            
    return data_o, data_y, data_age, data_g

# --- NEW FUNCTION FOR COUNTING ---
def print_stratified_stats(set_name, labels, ages):
    """
    Counts and prints segments based on Age (Young/Old) and Class (Normal/Apnea).
    0 = Young, 1 = Old
    0 = Normal, 1 = Apnea
    """
    young_norm = 0
    young_apnea = 0
    old_norm = 0
    old_apnea = 0
    
    for lab, ag in zip(labels, ages):
        if ag == 0: # Young
            if lab == 0: young_norm += 1
            else: young_apnea += 1
        elif ag == 1: # Old
            if lab == 0: old_norm += 1
            else: old_apnea += 1
            
    total = len(labels)
    print(f"\n{'='*10} {set_name} STATISTICS {'='*10}")
    print(f"Total Segments: {total}")
    print(f"YOUNG (20-50): Normal = {young_norm:<6} | Apnea = {young_apnea:<6} | Subtotal = {young_norm + young_apnea}")
    print(f"OLD (>50):     Normal = {old_norm:<6} | Apnea = {old_apnea:<6} | Subtotal = {old_norm + old_apnea}")
    print("="*40 + "\n")

if __name__ == "__main__":
    os.makedirs(BASE_DIR_OUTPUT, exist_ok=True)

    # 1. Define Train/Test Split (Standard PhysioNet Split)
    # We use ALL subjects now, because we need mixed ages for adversarial training
    train_subs = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]
    
    test_subs = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]

    print(f"Training Subjects: {len(train_subs)}")
    print(f"Testing Subjects: {len(test_subs)}")

    # 2. Load Labels
    labels_abc = parse_answers_file("summary-training-2")
    labels_x = parse_answers_file("event-2-answers")
    all_labels = {**labels_abc, **labels_x}

    # 3. Run Extraction
    print("\nProcessing Training Data...")
    o_train, y_train, age_train, g_train = run_processing(train_subs, "Train Set", all_labels)
    
    print("\nProcessing Testing Data...")
    o_test, y_test, age_test, g_test = run_processing(test_subs, "Test Set", all_labels)

    print_stratified_stats("TRAINING SET", y_train, age_train)
    print_stratified_stats("TESTING SET", y_test, age_test)

    # 4. Save to Pickle
    save_path = os.path.join(BASE_DIR_OUTPUT, "adversarial_data.pkl")
    data_dict = {
        "o_train": o_train, "y_train": y_train, "age_train": age_train, "groups_train": g_train,
        "o_test": o_test,   "y_test": y_test,   "age_test": age_test,   "groups_test": g_test
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=2)

    print(f"\nDone! Data saved to {save_path}")