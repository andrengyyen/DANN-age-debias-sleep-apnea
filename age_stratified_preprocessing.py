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
# Folder containing the .dat and .hea files for a*, b*, c*, x*
BASE_DIR_DATA = "apnea-ecg-database-1.0.0" 
BASE_DIR_OUTPUT = "dataset"

# Signal Properties
FS = 100 
SAMPLES_PER_MIN = FS * 60
MARGIN_BEFORE = 2 
MARGIN_AFTER = 2 

# Age Stratification Settings
AGE_TRAIN_MIN = 51 # change this to 20 if you want to train on young
AGE_TRAIN_MAX = 90 # change this to 50 if you want to train on young
AGE_TEST_MIN = 20 # change this to 51 if you want to train on old
AGE_TEST_MAX = 50 # change this to 90 if you want to train on old

# Workers
NUM_WORKERS = 35 if cpu_count() > 35 else cpu_count() - 1

def parse_answers_file(file_path):
    """
    Parses the text files (event-2-answers, summary-training-2) 
    that contain minute-by-minute labels string blocks.
    Returns: Dictionary { 'record_name': ['N', 'N', 'A', ...] }
    """
    answers = {}
    answer_file_path = os.path.join(BASE_DIR_OUTPUT, file_path)
    if not os.path.exists(answer_file_path):
        print(f"Warning: {answer_file_path} not found.")
        return answers
        
    with open(answer_file_path, "r") as f:
        # Split by double newline to separate records
        blocks = f.read().strip().split("\n\n")
        
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines: continue
        
        # First line is the record name
        record_name = lines[0].strip()
        
        # Remaining lines contain the labels. 
        # Format: "0 NNNNN..." -> We split to ignore the index number
        full_label_str = ""
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                # parts[0] is the row index, parts[1] is the string of labels
                full_label_str += parts[1]
        
        answers[record_name] = list(full_label_str)
        
    return answers

def get_stratified_subjects(info_file):
    """
    Reads additional-information.txt to sort subjects into Train/Test based on Age.
    """
    train_recs = []
    test_recs = []
    ages = {}
    info_file_path = os.path.join(BASE_DIR_DATA, info_file)
    with open(info_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Valid lines start with record ID (a.., b.., c.., x..)
            if len(parts) > 8 and parts[0][0] in ['a', 'b', 'c', 'x']:
                try:
                    name = parts[0]
                    # Based on file structure: Record... [8]Age
                    age = int(parts[8]) 
                    ages[name] = age
                    
                    if AGE_TRAIN_MIN <= age <= AGE_TRAIN_MAX:
                        train_recs.append(name)
                    elif AGE_TEST_MIN <= age <= AGE_TEST_MAX:
                        test_recs.append(name)
                except ValueError:
                    continue
                    
    return train_recs, test_recs, ages

def balance_groups(group1, group2):
    """
    Downsamples the larger group so both groups have the same number of subjects.
    """
    target_count = min(len(group1), len(group2))
    
    # Shuffle first to ensure random selection
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(group1)
    random.shuffle(group2)
    
    # Slice both to the target count
    balanced_1 = group1[:target_count]
    balanced_2 = group2[:target_count]
    
    return balanced_1, balanced_2

def process_ecg_file(file_name, labels):
    """
    Extracts 5-minute windows and converts to Features (RR-Interval + Amplitude).
    Identical logic to original author's code.
    """
    try:
        X_data = []
        y_labels = []
        group_names = []

        # Read Signal
        record_path = os.path.join(BASE_DIR_DATA, file_name)
        signals = wfdb.rdrecord(record_path, channels=[0]).p_signal[:, 0]

        for j in range(len(labels)):
            if j < MARGIN_BEFORE or (j + 1 + MARGIN_AFTER) > len(signals) / float(SAMPLES_PER_MIN):
                continue

            # Windowing
            start_idx = int((j - MARGIN_BEFORE) * SAMPLES_PER_MIN)
            end_idx = int((j + 1 + MARGIN_AFTER) * SAMPLES_PER_MIN)
            segment = signals[start_idx:end_idx]

            # Filtering
            segment, _, _ = st.filter_signal(segment, ftype='FIR', band='bandpass', 
                                             order=int(0.3 * FS), frequency=[3, 45], sampling_rate=FS)

            # R-Peak Detection
            rpeaks, = hamilton_segmenter(segment, sampling_rate=FS)
            rpeaks, = correct_rpeaks(segment, rpeaks=rpeaks, sampling_rate=FS, tol=0.1)

            # Validity Checks
            total_minutes = 1 + MARGIN_AFTER + MARGIN_BEFORE
            if len(rpeaks) / total_minutes < 40 or len(rpeaks) / total_minutes > 200:
                continue

            # Feature Extraction
            rri_time = rpeaks[1:] / float(FS)
            rri_values = medfilt(np.diff(rpeaks) / float(FS), kernel_size=3)
            ampl_time = rpeaks / float(FS)
            ampl_values = segment[rpeaks]

            hr = 60 / rri_values
            if np.all(np.logical_and(hr >= 20, hr <= 300)):
                X_data.append([(rri_time, rri_values), (ampl_time, ampl_values)])
                y_labels.append(0. if labels[j] == 'N' else 1.)
                group_names.append(file_name)

        return X_data, y_labels, group_names

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return [], [], [] # Return empty on error to keep loop running

if __name__ == "__main__":
    os.makedirs(BASE_DIR_OUTPUT, exist_ok=True)

    # 1. Stratify Subjects by Age
    print("Stratifying subjects by age...")
    train_subs, test_subs, age_map = get_stratified_subjects("additional-information.txt")
    print(f"Training ({AGE_TRAIN_MIN}-{AGE_TRAIN_MAX}): {len(train_subs)} subjects")
    print(f"Testing ({AGE_TEST_MIN}-{AGE_TEST_MAX}):    {len(test_subs)} subjects")

    # 2. Balance Groups - Uncomment this if you want balance number of subject in two age groups
    # print("Balancing groups...")
    # train_subs, test_subs = balance_groups(train_subs, test_subs)
    # print(f"Balanced counts -> Train: {len(train_subs)}, Test: {len(test_subs)}")

    # 3. Load Labels
    # 'summary-training-2' contains labels for a*, b*, c*
    # 'event-2-answers' contains labels for x*
    labels_abc = parse_answers_file("summary-training-2")
    labels_x = parse_answers_file("event-2-answers")
    
    # Merge dictionaries
    all_labels = {**labels_abc, **labels_x}

    # 4. Processing Function
    def run_processing(subject_list, dataset_name):
        data_o, data_y, data_g = [], [], []
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for name in subject_list:
                if name in all_labels:
                    futures[executor.submit(process_ecg_file, name, all_labels[name])] = name
                else:
                    print(f"Missing labels for {name}, skipping.")

            for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
                X, y, g = future.result()
                data_o.extend(X)
                data_y.extend(y)
                data_g.extend(g)
        return data_o, data_y, data_g

    # 5. Run Extraction
    print("\nProcessing Training Data...")
    o_train, y_train, g_train = run_processing(train_subs, "Train Set")
    
    print("\nProcessing Testing Data...")
    o_test, y_test, g_test = run_processing(test_subs, "Test Set")

    # 6. Save to Pickle
    save_path = os.path.join(BASE_DIR_OUTPUT, "stratified_data.pkl")
    data_dict = {
        "o_train": o_train, "y_train": y_train, "groups_train": g_train,
        "o_test": o_test,   "y_test": y_test,   "groups_test": g_test, "age_map": age_map
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=2)

    print(f"\nDone! Data saved to {save_path}")