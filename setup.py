from setuptools import setup, find_packages

setup(
    name="healthcare_project",
    version="0.1",
    description="A healthcare data predicting stroke occurrence project.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "seaborn",
        "matplotlib",
        "pyarrow", 
        "glum",
    ],
    python_requires=">=3.8",
)
