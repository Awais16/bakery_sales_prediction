# Sales Forecasting for a Bakery Branch

## Team Members (Raum#5)
- Awais 
- Yousra

## Repository Link

https://github.com/Awais16/bakery_sales_prediction

## Description

This project focuses on sales forecasting for a bakery branch, utilizing historical sales data spanning from July 1, 2013, to July 30, 2018, to inform inventory and staffing decisions. We aim to predict future sales for six specific product categories: Bread, Rolls, Croissants, Confectionery, Cakes, and Seasonal Bread. Our methodology integrates statistical and machine learning techniques, beginning with a baseline linear regression model to identify fundamental trends, and progressing to a sophisticated neural network designed to discern more nuanced patterns and enhance forecast precision. The initiative encompasses data preparation, crafting bar charts with confidence intervals for visualization, and fine-tuning models to assess their performance on test data from August 1, 2018, to July 30, 2019, using the Mean Absolute Percentage Error (MAPE) metric for each product category.

### Task Type
Regression
Neural Net

### Results Summary

-   **Best Model:** Neural Network ([check model readme](3_Model/README.md))
-   **Evaluation Metric:** MAPE
-   **Result by Category** (Identifier):
    -   **Bread** (1): 18.7823%
    -   **Rolls** (2): 13.5508%
    -   **Croissant** (3): 17.2352%
    -   **Confectionery** (4): 21.9707%
    -   **Cake** (5): 14.0129%
    -   **Seasonal Bread** (6): 36.9580%

## Documentation

1.  [**Data Import and Preparation**](0_DataPreparation/)
3.  [**Dataset Characteristics (Barcharts)**](1_DatasetCharacteristics/)
4.  [**Baseline Model**](2_BaselineModel/)
5.  [**Model Definition and Evaluation**](3_Model/)
6.  [**Presentation**](4_Presentation/README.md)

## Getting Started

1. Clone the repository.
2. Add your data files to the `data/` directory.
3. Open and run the notebooks in your preferred environment.

## Git Workflow
We are following trunk base workflow for git
- Git checkout main
- Pull main branch: git pull main 
- Create a feature branch: git branch -b feat/new_feat
- Commit Changes and push to your branch
- create a pull request from your branch (feat/new_feat) to main
- ** Clear All outputs ** before committing. (For now it will be great, we can setup nbstripout later)

## Kaggle Competition link
[Kaggle link](https://www.kaggle.com/competitions/bakery-sales-prediction-winter-2025-26)

## Cover Image

![](CoverImage/cover_image.png)
