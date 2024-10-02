# Banking Fraud Detection

## Project Overview
This project focuses on detecting fraudulent transactions in banking systems using machine learning models. With the growing complexity of fraud techniques, traditional rule-based systems are often insufficient. To address this, we utilize advanced machine learning algorithms, including XGBoost, to detect fraudulent behavior in transaction data. 

The model has been optimized with techniques such as **SMOTE** and **ADASYN** to handle data imbalance, ensuring a high performance in detecting fraud cases. We also compare the performance of multiple models like **KNN**, **Artificial Neural Networks (ANN)**, and **XGBoost**, with **XGBoost** emerging as the best model with an **ROC AUC score of 93.97** and a **precision of 96.8%**.

## Features
- **Data Cleaning and Preprocessing**: Includes handling missing values, feature extraction, and normalization.
- **Model Comparison**: Multiple models including XGBoost, KNN, and ANN were compared and evaluated for fraud detection.
- **Imbalanced Data Handling**: Techniques such as SMOTE and ADASYN were applied to manage the imbalance in fraudulent and non-fraudulent transactions.
- **Model Optimization**: Hyperparameter tuning was performed to enhance model performance.

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning model building.
- **XGBoost**: For building the fraud detection model.
- **SMOTE/ADASYN**: For handling data imbalance.
- **Matplotlib & Seaborn**: For data visualization.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains credit card transactions made by European cardholders in September 2013, with fraudulent transactions accounting for 0.172% of the total transactions. The dataset is highly imbalanced, requiring techniques such as SMOTE and ADASYN for better model training.

## Future Improvements
Implement real-time fraud detection by integrating the model with a streaming system.
Explore deep learning approaches such as LSTM for sequence-based anomaly detection.
Further tune models and experiment with ensemble techniques to improve accuracy.
