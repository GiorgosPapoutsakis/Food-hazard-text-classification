# Food-hazard-text-classification

## Overview

This repository contains a project developed as part of the Applied Machine Learning course. The project is based on the SemEval 2025 Task 9: The Food Hazard Detection Challenge, which focuses on developing explainable classification models for food-related incident reports collected from the web.

## Task Description

The challenge consists of two sub-tasks:

- **Text Classification (ST1):** Predicting the type of food hazard and product category from text.
- **Vector Detection (ST2):** Identifying exact hazard and product labels.

The dataset includes structured information such as date, country, title, and full-text recall notices, with heavily imbalanced class distributions across multiple categories.

## Technologies Used

For this project, various machine learning and natural language processing (NLP) techniques were applied:

- **Data Processing:** Used `pandas` for handling datasets.
- **Feature Engineering:** Applied TF-IDF (from `scikit-learn`) to transform text into numerical features.
- **Machine Learning Models:** Utilized Support Vector Machines (LinearSVC) for classification.
- **Text Preprocessing:** Used `nltk` for tokenization, stopword removal, and lemmatization.
- **Model Evaluation:** Employed *ross-validation and F1-score to assess performance.
- **Hyperparameter Optimization:** Used `Optuna` to optimize model parameters.
- **Development Environment:** Implemented using Python in Jupyter Notebooks.

## Methodology

### Preprocessing

- Tokenization, stopword removal, and lemmatization using `nltk`.
- Conversion of text into numerical representations using TF-IDF.

### Modeling

- Trained a Support Vector Machine (LinearSVC) for classification.
- Used a `scikit-learn` pipeline for feature transformation and modeling.

### Evaluation

- Applied cross-validation to ensure model generalization.
- Used macro F1-score to handle class imbalance.
- Fine-tuned model parameters using Optuna.
