# Age-Specific Bias Mitigation in OSA Detection (SIT723/SIT729 Deakin University)

This repository contains the implementation of a **Domain-Adversarial Neural Network (DANN)** designed to mitigate age-specific bias in deep learning-based obstructive sleep apnea (OSA) detection using single-lead ECG signals.

## Project Details

| **Field** | **Details** |
| :--- | :--- |
| **Unit** | SIT723 Research Techniques & Applications / SIT792 Minor Thesis |
| **Institution** | Deakin University |
| **Student** | [Trung Hoang Anh (Andre) Nguyen](https://www.linkedin.com/in/andre-nguyen-0298a9287/) |
| **Supervisor** | [Dr. Md. Ahsan Habib](https://experts.deakin.edu.au/50940-ahsan-habib) |

---

## Core Methodology

The research focuses on resolving **Simpson's Paradox** in automated sleep medicine—a phenomenon where high aggregate accuracy masks significant diagnostic failures in specific age subgroups.

This implementation utilizes a hybrid **CNN-Transformer-LSTM** backbone integrated with a **Gradient Reversal Layer (GRL)** and a shared bottleneck layer. This architecture forces the model to learn age-invariant features, ensuring that the diagnostic performance remains consistent across different demographic cohorts.

---

## Repository Structure

* `preprocessing.py`
    * Handles the standardization of the raw PhysioNet Apnea-ECG signals.
    * Includes R-peak detection via the **Hamilton algorithm** and cubic spline interpolation.

* `age_stratified_preprocessing.py`
    * Processes and partitions the dataset into two primary demographic cohorts:
        * **Young:** $\le 50$ years
        * **Old:** $> 50$ years

* `DANN.py`
    * Contains the dual-head architecture featuring a primary **apnea task classifier** and a secondary **adversarial age discriminator**.

* `age_bias_testing.py`
    * Script to run cross-age validation experiments (e.g., training on Young and testing on Old) to check the sensitivity and specificity of the model on different age groups".

* `/apnea-ecg-database-1.0.0`
    * An empty directory intended for the **PhysioNet Apnea-ECG Database**.
    * **Note:** Users must download the data from PhysioNet to this folder before running scripts.

* `/dataset`
    * An empty directory intended for the output from preprocessing

---

## References

1.  **Baseline Model:** Pham, D.T., & Mouček, R. (2025). Efficient sleep apnea detection using single-lead ECG with CNN-Transformer-LSTM. *Computers in Biology and Medicine*.
2.  **DANN Theory:** Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*.
3.  **Bottleneck Framework:** Chen, X., et al. (2023). BAFNet: Bottleneck attention based fusion network for sleep apnea detection. *IEEE JBHI*.
