# Blood–Brain Barrier (BBB) Permeability Prediction

This repository presents a comprehensive study on the prediction of **Blood–Brain Barrier (BBB) permeability** of chemical compounds using advanced machine learning and deep learning methodologies. The work is aimed at facilitating drug discovery for central nervous system (CNS) therapeutics by identifying compounds capable of crossing the BBB.

---

## Dataset

* **Primary Source:** \[Blood–Brain Barrier Database (B3DB)]\(https://github.com/theochem/B3DB/tree/main/B3DB)
* **Available Data (Google Drive):** [Access Dataset](https://drive.google.com/drive/folders/18pLes5eUOvhtCCu6JOllBcqHQFMWuLHZ?usp=sharing)
* **Total Compounds:** 7,807
* **Class Distribution:**

  * BBB-permeable (BBB+): 63.5%
  * Non-permeable (BBB−): 36.5%

### Descriptor Generation

* Molecular descriptors were calculated using **Mordred v1.2.0**.
* Chemical structures were processed in **RDKit v2022.09.5**.
* The initial feature matrix consisted of **7,807 compounds × 1,613 descriptors**.

---

## Data Preprocessing

* **Molecular weight filtering:** Retained molecules within 150–800 Da (drug-like range).
* **Feature cleaning:**

  * Removed descriptors with more than 5% missing values.
  * Imputed remaining missing values with the median.
  * Discarded features with variance < 0.01.
* **Outlier treatment:** Feature values were capped at the 1st and 99th percentiles.

After preprocessing, the dataset contained **7,298 compounds × 1,613 descriptors**.

---

## Feature Selection

A two-stage feature selection pipeline was applied:

1. **Distance correlation filtering** – reduced to 292 descriptors.
2. **Spearman correlation filtering (ρ > 0.85)** – final feature set of 59 descriptors.

Representative features included: *TopoPSA, Lipinski, GhoseFilter, SLogP, nHBDon, and selected ATSC/GATS descriptors*.

---

## Model Development

The following models were trained and evaluated:

* **Tree-based models:** Random Forest, Gradient Boosting, XGBoost, Extra Trees, LightGBM, CatBoost
* **Linear and kernel-based models:** Logistic Regression, Support Vector Classifier (SVC)
* **Graph and deep learning approaches:**

  * Graph Convolutional Network (GCN)
  * ChemBERTa embeddings combined with LightGBM
  * Fine-tuned ChemBERTa

---

## Results

* **Best Performing Model:** CatBoost

  * AUC: 0.9570
  * Accuracy: 0.9027
  * MCC: 0.790
  * F1 Score: 0.9238

* **Other high-performing models:** LightGBM and XGBoost (F1 > 0.91).

* **Deep learning approaches (ChemBERTa-based):** Achieved competitive performance with descriptor-based models (F1 ≈ 0.89).

---

## Rule-Based Interpretability

A decision tree analysis was conducted using **Lipinski** and **Ghose** rule-based filters:

* Compounds passing both rules: 74% BBB+
* Compounds failing both rules: 68% BBB−

This demonstrates that traditional drug-likeness rules remain meaningful for distinguishing BBB permeability.

---

## Repository Structure

```
├── Predict_BBB/                  # Raw and processed datasets  
├── all notebooks             # All Jupyter notebooks 
└── README.md              # Project documentation  
```

---

## Future Directions

* Enhanced fine-tuning of **ChemBERTa** for descriptor-free learning.
* Incorporation of **external validation datasets**.
* Deployment of a **web-based prediction tool** for real-time BBB permeability assessment.

---

## Citation

If this work is used in academic or research projects, please cite the following resources:

* **B3DB:** Blood–Brain Barrier Database
* **Mordred and RDKit:** Descriptor calculation and cheminformatics tools
* **Machine Learning Frameworks:** CatBoost, LightGBM, XGBoost
