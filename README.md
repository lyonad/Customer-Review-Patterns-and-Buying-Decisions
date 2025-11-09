# Comparative Analysis of Gradient Boosting Models for Customer Review Classification with Grey Wolf Optimization

## Overview

This project implements a comparative analysis of three popular gradient boosting algorithms - XGBoost, LightGBM, and CatBoost - optimized using the Grey Wolf Optimization (GWO) algorithm for customer review classification. The goal is to predict customer purchase decisions based on demographic data, reviews, and other customer attributes.

## DOI and Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17562401.svg)](https://doi.org/10.5281/zenodo.17562401)

**Citation (APA):**
```
Lyon Ambrosio Djuanda. (2025). lyonad/Customer-Review-Patterns-and-Buying-Decisions: Initial Release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17562401
```

## Table of Contents
- [Project Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Grey Wolf Optimization](#grey-wolf-optimization)
- [Results](#results)
- [Files Structure](#files-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is the "Customer Review Patterns and Buying Decisions" dataset from Kaggle, which contains:
- **Records:** 100 customer entries
- **Features:**
  - Serial Number (ID)
  - Age (18-59)
  - Gender (Male/Female)
  - Review (Poor/Average/Good)
  - Education (School/UG/PG)
  - Purchased (Yes/No) - Target variable

The dataset has no missing values and is 100% complete. It captures customer feedback along with demographic data to study how personal factors influence purchase behavior.

## Methodology

### Feature Engineering
- Created age groups: Young (18-25), Adult (26-35), Middle (36-50), Senior (51-60)
- Created feature combinations: Gender_Review, Review_Edu
- Applied StandardScaler to numerical features
- Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset

### Models Analyzed
- **XGBoost:** Regularized gradient boosting with tree pruning
- **LightGBM:** Gradient boosting framework using histogram-based algorithms
- **CatBoost:** Gradient boosting that handles categorical features automatically

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Grey Wolf Optimization

Grey Wolf Optimizer (GWO) is a metaheuristic algorithm inspired by the leadership hierarchy and hunting behavior of grey wolves. In this project, GWO is used to optimize hyperparameters for each gradient boosting model:

- **XGBoost:** Optimizes n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda
- **LightGBM:** Optimizes n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda
- **CatBoost:** Optimizes iterations, depth, learning_rate, reg_lambda

The optimization process involves simulating the social hierarchy of wolves (alpha, beta, delta) to find optimal parameter configurations.

## Results

The models showed the following performance after GWO optimization:

| Model      | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| CatBoost   | 86.96%   | 89.57%    | 86.96% | 86.65%   |
| XGBoost    | 82.61%   | 83.51%    | 82.61% | 82.41%   |
| LightGBM   | 73.91%   | 74.47%    | 73.91% | 72.88%   |

**Best Model:** CatBoost achieved the highest accuracy of 86.96%

## Files Structure

```
├── customer_review_gwo_optimized.ipynb       # Jupyter notebook with complete analysis
├── customer_review_gwo_optimized.py          # Python script implementation
├── data/
│   ├── Customer_Review (1).csv               # Customer review dataset
│   └── DATASET_INFO.md                       # Dataset documentation
├── results/
│   ├── accuracy_comparison.png               # Accuracy comparison visualization
│   ├── best_model_catboost.pkl               # Trained best model (CatBoost)
│   ├── best_model_xgboost.pkl                # Trained XGBoost model
│   ├── confusion_matrix_catboost.png         # Confusion matrix for CatBoost
│   ├── confusion_matrix_xgboost.png          # Confusion matrix for XGBoost
│   ├── customer_review_analysis_report.txt   # Detailed analysis report
│   ├── feature_importance_catboost.png       # Feature importance for CatBoost
│   ├── feature_importance_xgboost.png        # Feature importance for XGBoost
│   └── ...
├── catboost_info/                            # CatBoost training logs
└── README.md                                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Customer-Review-Patterns-and-Buying-Decisions
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

If no requirements.txt exists, install the necessary packages:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn joblib
```

## Usage

### Jupyter Notebook
1. Start Jupyter notebook:
```bash
jupyter notebook
```
2. Open `customer_review_gwo_optimized.ipynb`
3. Run all cells to reproduce the analysis

### Python Script
Run the Python script directly:
```bash
python customer_review_gwo_optimized.py
```

### Custom Analysis
To use your own dataset:
1. Place your CSV file in the `data/` directory
2. Update the file path in the script/notebook
3. Ensure your dataset has the same column structure or modify the feature engineering section accordingly

## Dependencies

The project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- imbalanced-learn (for SMOTE)
- matplotlib
- seaborn
- joblib (for model serialization)
- warnings (standard library)

## Key Features

1. **Automated Hyperparameter Optimization:** Uses Grey Wolf Optimization to find optimal parameters for each model
2. **Comprehensive Model Comparison:** Evaluates three state-of-the-art gradient boosting algorithms
3. **Feature Engineering:** Automatically creates meaningful feature combinations
4. **Data Augmentation:** Uses SMOTE to handle imbalanced datasets
5. **Visualization:** Generates comprehensive visualizations of model performance
6. **Model Persistence:** Saves the best performing model for future use

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Improvements

- Implement additional ensemble methods
- Explore deep learning approaches
- Add more advanced feature engineering techniques
- Implement cross-validation for more robust evaluation
- Extend to multi-class classification if the dataset is expanded

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Dataset source: [Kaggle - Customer Review Patterns and Buying Decisions](https://www.kaggle.com/datasets/ayeshaimran123/customer-review-patterns-and-buying-decisions)
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Microsoft LightGBM: A highly efficient gradient boosting decision tree framework
- Prokhorenkova, L., Gusev, G., & Vorobev, A. (2018). CatBoost: unbiased boosting with categorical features

## Acknowledgments

- Thanks to Ayesha Imran for providing the customer review dataset
- The Grey Wolf Optimization algorithm implementation was adapted from research literature
- Special thanks to the open-source community for providing the gradient boosting libraries