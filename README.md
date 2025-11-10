# Customer Review Patterns and Buying Decisions

## Research Title
**Predicting Purchase Likelihood using Advanced Gradient Boosting XGBoost LightGBM and CatBoost and Comprehensive Customer Data Reviews and Demographics**

A machine learning project that analyzes customer review patterns and predicts buying decisions by comparing three gradient boosting algorithms: XGBoost, LightGBM, and CatBoost.

## Project Overview

The goal is to understand how customer demographics (age, gender, education) and review quality influence purchasing behavior. The project uses a dataset from Kaggle containing customer feedback and purchase decisions.

## Files

- `model_comparison.py` - Main script for model comparison using default hyperparameters
- `model_comparison.ipynb` - Jupyter notebook version with the same functionality and interactive execution
- `data/Customer_Review (1).csv` - Raw dataset
- `data/DATASET_INFO.md` - Dataset documentation
- `results/` - Generated reports, metrics, and visualizations (created after running the script)
- `LICENSE` - Project license

## Dataset

- **Source**: Kaggle - Customer Review Patterns and Buying Decisions
- **Size**: 100 samples (augmented to ~112 with SMOTE)
- **Features**:
  - Age (18-59)
  - Gender (Male/Female)
  - Review quality (Poor/Average/Good)
  - Education level (School/UG/PG)
  - Purchased (Yes/No) - Target variable

## Model Comparison Results

Recent results using optimized default hyperparameters:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 60.87% | 0.612 | 0.609 | 0.600 |
| LightGBM | 73.91% | 0.769 | 0.739 | 0.729 |
| CatBoost | **78.26%** | 0.801 | 0.783 | 0.778 |

**Best Model**: CatBoost with 78.26% accuracy

## Usage

### Option 1: Run the Python Script
```bash
python model_comparison.py
```

### Option 2: Use the Jupyter Notebook
Open `model_comparison.ipynb` in Jupyter and run all cells sequentially for an interactive experience.

Both options perform the same analysis:
- Loads and preprocesses the customer review dataset
- Trains XGBoost, LightGBM, and CatBoost models with optimized default parameters
- Evaluates and compares model performance using multiple metrics
- Generates comprehensive visualizations and reports

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- imbalanced-learn (for SMOTE)
- matplotlib
- seaborn

## Key Findings

1. **CatBoost** performs best with optimized default parameters, achieving 78.26% accuracy
2. **LightGBM** shows strong performance at 73.91% accuracy
3. **XGBoost** performs moderately at 60.87% accuracy with default settings
4. Feature engineering (age groups, categorical combinations) improves model performance
5. SMOTE effectively balances the dataset for better training

## Generated Outputs

After running either the Python script or the Jupyter notebook, the following files are created in the `results/` folder:

- `results/model_comparison_report.txt` - Detailed analysis report
- `results/model_comparison_metrics.csv` - Model performance metrics
- `results/model_accuracy_comparison.png` - Accuracy comparison chart
- `results/model_comparison_metrics.png` - All metrics comparison chart
- `results/confusion_matrices.png` - Confusion matrices for all models
- Saved model files (PKL format) for each trained model