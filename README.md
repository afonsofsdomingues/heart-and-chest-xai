<div align="center">
      
# Heart-and-Chest-XAI

</div>

## Overview
This project investigates interpretable and explainable machine learning methods for medical classification tasks using both tabular and imaging data. In Part 1, we focus on the heart disease prediction task, applying shallow models like logistic regression with L1 regularization and deep models such as MLPs and NAMs, using SHAP for post-hoc explanation. In Part 2, we analyze chest X-ray images for pneumonia detection with CNNs, employing saliency-based methods including Integrated Gradients and Grad-CAM to highlight relevant image regions. Our findings offer insights into selecting suitable, trustworthy, and interpretable models for practical medical applications.

For details, refer to our [📄 Project Report](./Machine_Learning_for_Health_Care___Project2_Report.pdf).

You can also find the [📝 Project Handout](./Project2_Explainability_and_Interpretability_Handout.pdf).

We achieved a grade of **5.8**/6 for this project.

## Dataset – Part 1: Heart Disease Prediction

- **Source**: [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (aggregated from the UCI Machine Learning Repository)  
- **Data type**: Tabular (clinical and demographic records)  
- **Features**:  
  - **11 variables**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope  
- **Target**: Binary classification — **0: Normal, 1: Heart Disease**  

## Dataset – Part 2: Pneumonia Prediction

- **Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Data type**: Medical images (grayscale JPEG)  
- **Input**: Chest X-ray images of varying resolution.
- **Target**: Binary classification — **Normal**, **Pneumonia**

## How to Run the Project

1. **Set Up the Environment**:
      1. **Part 1 (Heart Disease Prediction)**:
          - Have Python 3.8 or higher installed.
          - Create a virtual environment:
            ```bash
            python -m venv venv_part1
            source venv_part1/bin/activate  # On Windows: venv_part1\Scripts\activate
            ```
          - Install dependencies:
            ```bash
            pip install -r requirements_1.txt
            ```

      2. **Part 2 (Pneumonia Prediction)**:
          - Create another virtual environment:
            ```bash
            python -m venv venv_part2
            source venv_part2/bin/activate  # On Windows: venv_part2\Scripts\activate
            ```
          - Install dependencies:
            ```bash
            pip install -r requirements_2.txt
            ```

2. **Prepare the Datasets**:

    For **Part 1**
    - Download the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
    - Unzip the dataset and place its contents in the `heart_failure/` folder.

    The file structure for this part should look like:
    ```
    heart_failure/
    ├── test_split.csv
    ├── train_val_split.csv
    ```

    For **Part 2** (Pneumonia Detection):
    - Download the dataset from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
    - Unzip the dataset and place its contents in the `chest_xray/` folder.

    The file structure for this part should look like:
    ```
    chest_xray/
    ├── test/
    │   ├── NORMAL/
    │   │   ├── ...
    │   ├── PNEUMONIA/
    │       ├── ...
    ├── train/
    │   ├── NORMAL/
    │   │   ├── ...
    │   ├── PNEUMONIA/
    │       ├── ...
    ├── val/
    │   ├── NORMAL/
    │   │   ├── ...
    │   ├── PNEUMONIA/
    │       ├── ...
    ```

3. **Run the Notebooks**:
   - Execute the Jupyter notebooks:
     - `heart_failure.ipynb` for part 1
     - `pneumonia.ipynb` for part 2

## Folder Structure

```
heart-and-chest-xai/
├── chest_xray/             # Raw data for part 2
├── heart_failure/          # Raw data for part 1
├── nam/                    # Neural Additive Model for part 1
├── heart_failure.ipynb     # Notebook with solution for part 1
├── pneumonia.ipynb         # Notebook with solution for part 2
├── requirements_1.txt      # Python dependencies for part 1
├── requirements_2.txt      # Python dependencies for part 2
├── README.md               # Project documentation
└── transforms.py           # Utility functions for data augmentation (part 2)
```
