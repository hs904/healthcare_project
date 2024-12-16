# healthcare_project
Prediction of stroke occurrences using healthcare data.

---

## **Setup**
### **1. Create and activate the environment**
conda env create -f environment.yml
conda activate healthcare_project

### **2. Install the repo as a package:**
pip install -e .

### Run EDA_cleaning
1. Launch Jupyter Notebook: jupyter notebook
2. Navigate to the project directory and open eda_cleaning.ipynb.
3. Run the cells in the notebook sequentially to:
   Explore the raw dataset from data/raw_data/.
   Visualize distributions, correlations, and outliers.
   Perform cleaning steps like handling missing values and removing outliers.
   Save the cleaned dataset as prepared_data.parquet in the data/ directory.

### Run Model Training 
python model_training.py

