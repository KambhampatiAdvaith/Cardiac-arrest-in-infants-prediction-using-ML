# Cardiac-arrest-in-infants-prediction-using-ML
 Project Overview

This project, undertaken as part of an internship at IIIT Kottayam, focuses on developing a robust machine learning **methodology** for the early prediction of critical events using Electronic Health Records (EHR). The ultimate aim is to apply such a methodology to the challenging problem of **early prediction of pediatric cardiac arrest in infants.**

Due to the significant challenges in accessing specific, de-identified infant cardiac arrest EHR data, this initial phase concentrates on building and validating a comprehensive ML pipeline using a publicly available **adult cardiac patient dataset (`CardiacPatientData.csv`) as a proxy.** The insights and techniques developed here are intended to serve as a foundational blueprint for future application to infant-specific data.

## Key Objectives & Contributions

*   To develop an end-to-end machine learning pipeline for processing EHR-like data.
*   To implement robust techniques for handling common EHR data challenges, including:
    *   Missing data (various imputation strategies, including KNNImputer and missingness indicators).
    *   Class imbalance.
*   To perform feature engineering to derive clinically relevant predictors.
*   To train, evaluate, and compare multiple machine learning models for predictive performance.
*   To employ model interpretability techniques (SHAP analysis) to understand model behavior on the proxy dataset.
*   To develop a conceptual prototype dashboard demonstrating potential model application.
*   To critically discuss the limitations of using proxy data and outline the necessary adaptations for applying the developed methodology to the infant cardiac arrest prediction domain.
