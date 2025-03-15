# Hyperspectral Data Analysis for Vomitoxin Prediction

This project involves the analysis of hyperspectral data to predict vomitoxin (DON) levels in agricultural samples. The analysis includes comprehensive data preprocessing, feature engineering, dimensionality reduction, and machine learning model development.

## Project Overview

The goal of this project is to develop a predictive model for vomitoxin (deoxynivalenol or DON) concentration using hyperspectral data. Vomitoxin is a mycotoxin produced by Fusarium fungi that affects grain crops and poses health risks to humans and animals.

## Dataset

The analysis was performed on a dataset containing:
- 500 samples
- 1117 features (primarily spectral reflectance values)
- Target variable: vomitoxin concentration in parts per billion (ppb)
- Data split: 320 training samples, 80 validation samples, 100 test samples

## Key Components

### Data Preprocessing
- Missing value handling with column-wise mean imputation
- Outlier detection and treatment using IQR method (clipping values beyond 3× IQR)
- Target variable transformation using log transformation to address skewed distribution
- Feature engineering:
  - Spectral derivatives calculation (447 derivative features)
  - Band ratios computation for enhanced spectral relationships
  - Normalized difference indices (NDIs) for spectral feature enhancement
- Data normalization using robust scaling to handle remaining outliers

### Dimensionality Reduction
- Principal Component Analysis (PCA)
  - 10 principal components retained for analysis
  - Visualizations of 2D and 3D PCA projections
  - Feature importance analysis based on PCA loadings
  - Variance explained analysis
  
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - 3 t-SNE components retained for analysis
  - Clustering pattern analysis with non-linear mapping
  - 2D and 3D visualizations with target value coloring
  - Cluster analysis using silhouette scores

### Machine Learning Models
- Neural Network Models:
  - Standard Neural Network (4 hidden layers with 211 neurons each)
  - LSTM Model (47 LSTM units, 90 dense units)
  - CNN Model (32 filters, kernel size 3, 64 dense layers)
  - Attention-based Model for feature importance focus

### Model Training
- Custom early stopping implementation with patience of 10 epochs
- Gradient clipping with a maximum norm of 1.0
- Batch processing with size of 32 
- Adam optimizer with hyperparameter-optimized learning rates
- Comprehensive NaN handling in both training and inference
- GPU acceleration when available

### Hyperparameter Optimization
- Bayesian optimization for hyperparameter tuning
- 5-fold cross-validation for model stability assessment
- Optimization of key parameters:
  - Learning rates
  - Network architecture (layer sizes, number of layers)
  - Regularization parameters (dropout rates)
  - Model-specific parameters (filter sizes, kernel sizes)

### Model Evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)
- 5-fold Cross-validation for model stability
- Learning curves analysis to diagnose bias-variance tradeoff
- Actual vs. predicted value visualization

## Results

The analysis revealed that:
- Dimensionality reduction techniques effectively captured the variance in the hyperspectral data with PCA identifying key components
- Feature engineering significantly improved model performance, with derivative spectra providing valuable information
- Neural network models, particularly LSTM and attention-based models, showed strong performance in capturing complex spectral patterns
- The most important spectral regions for vomitoxin prediction were identified through feature importance analysis
- Log transformation of the target variable (vomitoxin concentration) improved model performance significantly
- Attention mechanism provided insights into which spectral regions were most relevant for vomitoxin prediction
- Robust scaling and gradient clipping were effective at maintaining model stability with spectral data
- Careful NaN handling in both training and inference was critical for real-world applications

## Project Structure

The project is implemented in a Jupyter notebook (ImagoAI_Assign.ipynb) with the following structure:
1. **Data Preprocessing**
   - Data loading and cleaning
   - Feature engineering
   - Target transformation
   - Data normalization and splitting
   
2. **Dimensionality Reduction**
   - PCA implementation and analysis
   - t-SNE implementation and analysis
   - Feature importance evaluation
   
3. **Model Development**
   - Standard Neural Network implementation
   - LSTM model development
   - CNN model architecture
   - Attention mechanism implementation
   
4. **Model Training**
   - Batch processing
   - Early stopping
   - Gradient clipping
   - NaN handling
   - GPU acceleration
   
5. **Hyperparameter Optimization**
   - Bayesian optimization setup
   - Model-specific parameter spaces
   - Cross-validation evaluation
   
6. **Evaluation and Visualization**
   - Performance metrics calculation
   - Learning curve analysis
   - Prediction visualization

### Results Directory
The `results` directory contains:
- PCA and t-SNE visualizations and data files
- Model training visualizations (loss and MAE vs. epoch)
- Model performance visualizations (actual vs. predicted values)
- Metrics summary

## Dependencies
```
numpy==1.24.3
pandas==2.0.2
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
torch==2.0.1
scipy==1.10.1
```

## Future Work

- Implement advanced feature engineering techniques like Savitzky-Golay filtering and wavelet transforms
- Explore transfer learning approaches with pre-trained models on spectral data
- Develop ensemble methods combining predictions from multiple model architectures
- Enhance model interpretability through techniques like SHAP values and attention visualization
- Implement spectral data augmentation to expand the training dataset
- Optimize models for deployment in field conditions
- Integrate additional data types like weather and field conditions
- Implement advanced cross-validation strategies
- Add prediction uncertainty quantification
- Improve pipeline efficiency with parallel processing and caching

## Technical Report

A comprehensive technical report is available in [Technical_Report.md](Technical_Report.md), which includes:
- Detailed explanation of preprocessing steps and rationale
- In-depth analysis of dimensionality reduction insights
- Comprehensive coverage of model selection, training, and evaluation
- Key findings and detailed suggestions for improvement 