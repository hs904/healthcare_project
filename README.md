# healthcare_project
Prediction of stroke occurrences using healthcare data.

---

## **Setup**
### **1. Create and activate the environment**
conda env create -f environment.yml
conda activate healthcare_project

### **2. Install the repo as a package:**
pip install -e.

---
### Dataset
The raw dataset (`raw_data.csv`) is included in the repository under the `data/raw_data/` directory. No additional steps are required to obtain it.

### Run EDA_cleaning
jupyter notebook eda_cleaning.ipynb

### Run Model Training 
python model_training.py

