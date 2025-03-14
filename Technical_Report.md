# Technical Report: Hyperspectral Data Analysis for Vomitoxin Prediction

## 1. Introduction

This report details the development of a machine learning pipeline for predicting vomitoxin (deoxynivalenol or DON) concentration in agricultural samples using hyperspectral data. Vomitoxin is a mycotoxin produced by Fusarium fungi that affects grain crops and poses significant health risks to humans and animals when consumed in contaminated food products.

The project leverages advanced data preprocessing techniques, feature engineering, dimensionality reduction, and machine learning models to create an accurate prediction system from spectral reflectance data.

## 2. Preprocessing Steps and Rationale

### 2.1 Data Cleaning

The initial preprocessing steps focused on preparing the hyperspectral data for analysis:

- **Dataset Overview**: The analysis was performed on a dataset containing 500 samples with 1117 features, primarily consisting of spectral reflectance values across different wavelengths.

- **Column Name Standardization**: Column names were standardized to lowercase with underscores replacing spaces and hyphens, ensuring consistent naming conventions throughout the analysis.

- **Duplicate Removal**: Duplicate samples were identified and removed to prevent data leakage between training and testing sets and to avoid biasing the model.

- **Missing Value Handling**: Missing values were addressed using column-wise mean imputation. This approach was chosen over global mean imputation to preserve the spectral characteristics of each wavelength band. Column-wise imputation maintains the unique distribution of each spectral channel, which is crucial for hyperspectral data analysis.

- **Outlier Detection and Treatment**: The Interquartile Range (IQR) method was employed to identify and handle outliers. Values beyond 3 times the IQR from the quartiles were clipped rather than removed, preserving the overall sample size while reducing the impact of extreme values. This approach was critical for maintaining the integrity of the dataset while minimizing the influence of potentially erroneous measurements.

### 2.2 Feature Engineering

Several feature engineering techniques were applied to extract meaningful information from the raw spectral data:

- **Spectral Derivatives**: First-order derivatives were calculated between adjacent spectral bands, resulting in 447 derivative features. These derivatives capture the rate of change in reflectance across the spectrum and are often more informative than absolute reflectance values for detecting subtle spectral features. The derivatives highlight absorption and reflection transitions that are particularly relevant for detecting biochemical compounds like vomitoxin.

- **Band Ratios**: Ratios between selected spectral bands were computed, creating features that highlight relationships between different parts of the spectrum. These ratios can emphasize specific absorption or reflection characteristics related to the target variable. Band ratios also help normalize for illumination differences across samples, focusing on the relative differences between spectral regions.

- **Normalized Difference Indices (NDIs)**: Similar to common vegetation indices like NDVI, custom normalized difference indices were calculated between various band combinations. These indices enhance specific spectral responses while normalizing for overall reflectance intensity. NDIs were particularly useful for highlighting subtle differences in spectral signatures that might be associated with fungal infection and mycotoxin presence. The implementation carefully addressed potential division by zero issues by adding a small epsilon value (1e-6) to denominators.

### 2.3 Target Variable Transformation

The vomitoxin concentration (measured in ppb) was log-transformed using np.log1p() to address its right-skewed distribution. This transformation:
- Stabilized the variance, addressing heteroscedasticity issues
- Made the distribution more symmetric, improving the performance of models that assume normally distributed residuals
- Improved model performance by making the target variable more amenable to linear modeling techniques
- Handled the wide range of concentration values more effectively (spanning several orders of magnitude)
- Preserved zero values in the data by using log1p (log(1+x)) instead of direct logarithm

### 2.4 Data Normalization

Robust scaling was applied to the feature set, which:
- Centers the data around the median rather than the mean
- Scales based on the interquartile range instead of standard deviation
- Provides resilience against remaining outliers in the data
- Ensures all features contribute proportionally to the models regardless of their original scales
- Works particularly well with the non-parametric models used later in the analysis

The RobustScaler from scikit-learn was used for this purpose, which proved effective for handling the spectral data with potential remaining outliers after the IQR-based clipping.

### 2.5 Data Splitting

The dataset was divided into:
- Training set: 320 samples (64%)
- Validation set: 80 samples (16%)
- Test set: 100 samples (20%)

This splitting strategy provided sufficient data for model training while reserving adequate samples for validation and unbiased final evaluation. The train_test_split function from scikit-learn was used with a random_state of 42 to ensure reproducibility.

## 3. Insights from Dimensionality Reduction

### 3.1 Principal Component Analysis (PCA)

PCA was applied to reduce the high-dimensional spectral data while preserving the maximum amount of variance:

- The first 10 principal components were retained, capturing approximately 95% of the total variance in the dataset
- This significant dimensionality reduction (from over 1100 features to just 10) greatly improved computational efficiency while retaining most of the information
- The explained variance analysis showed that the first principal component alone captured about 45% of the total variance, indicating a strong primary pattern in the data
- The cumulative variance plot revealed that approximately 95% of the variance was captured by the first 10 components, providing a clear rationale for this choice of dimensionality
- The PCA results were saved as 'X_pca.npy' for further analysis and model integration
- Visualizations were created for both 2D and 3D representations of the PCA results, stored as 'pca_2d_visualization.png' and 'pca_3d_visualization.png'

Key insights from PCA:
- The high variance explained by the first component suggests a strong primary pattern in the spectral data
- The gradual decline in explained variance after the first few components indicates complex, multi-dimensional relationships in the data
- PCA loadings analysis revealed that certain spectral regions had disproportionate influence on the principal components, suggesting these regions contain important information for vomitoxin prediction
- Visualization of samples in PC space showed clustering patterns related to vomitoxin concentration levels, confirming the relationship between spectral characteristics and target variable
- The feature importance visualization (pca_feature_importance.png) highlighted the top 20 most influential spectral bands and derived features

The PCA implementation included careful handling of feature importance analysis based on component loadings, which identified the spectral regions most relevant to vomitoxin prediction. The top features based on total absolute loading included several derivative features and specific spectral bands, highlighting the importance of both raw spectral data and engineered features.

### 3.2 t-SNE Analysis

t-Distributed Stochastic Neighbor Embedding (t-SNE) was used to visualize the high-dimensional data in 2D and 3D spaces:

- t-SNE with 2 components was implemented first, providing a clear visualization of local data structure
- A 3D t-SNE analysis was also conducted for more comprehensive visualization
- Perplexity parameter was set to 30, balancing local and global structure preservation
- The t-SNE implementation included careful attention to random state initialization (random_state=42) for reproducibility

Cluster analysis on the t-SNE results revealed:
- Silhouette score analysis identified an optimal number of clusters in the data
- The clusters showed correlation with vomitoxin concentration levels
- The 2D t-SNE visualization with cluster coloring revealed natural groupings in the data
- The 3D visualization provided additional insights into the data structure, highlighting relationships that might be obscured in 2D

The combined insights from PCA and t-SNE proved invaluable for understanding the dataset structure:
- PCA identified the global variance structure and key feature relationships
- t-SNE revealed local neighborhood structures and potential subgroups in the data
- Together they provided complementary perspectives that informed the subsequent modeling approach

## 4. Model Selection, Training, and Evaluation

### 4.1 Model Selection and Architecture

Several neural network architectures were implemented to identify the most effective approach for hyperspectral data analysis:

- **Standard Neural Network**:
  - 4 hidden layers with 211 neurons per layer
  - Dropout regularization with a rate of 0.16
  - ReLU activation functions for hidden layers
  - Learning rate of 0.00028 optimized through Bayesian optimization
  - Implemented in PyTorch with careful gradient clipping to prevent exploding gradients

- **LSTM Model**:
  - Input reshaped to treat spectral bands as a sequence
  - 47 LSTM units in the recurrent layers
  - 90 units in the dense layer following LSTM processing
  - Dropout regularization with a rate of approximately 0.4
  - Learning rate of 0.00055 determined through optimization
  - Bidirectional LSTM layers to capture relationships in both directions of the spectral sequence

- **CNN Model**:
  - 1D convolutional architecture specifically designed for spectral data
  - 32 filters with kernel size 3 for feature extraction
  - MaxPooling layers to reduce dimensionality between convolutions
  - 64 units in the dense layer following convolutional processing
  - Dropout rate of 0.3 to prevent overfitting
  - Learning rate of 0.0005 optimized for stable convergence
  - Custom implementation to handle potential dimension mismatches

- **Attention Model**:
  - Custom attention mechanism to focus on the most relevant spectral regions
  - LSTM base layer with 64 units to process sequential spectral information
  - Attention layer to weight the importance of different spectral regions
  - Dense layer with 32 units for final prediction
  - Implementation based on a self-attention mechanism inspired by transformer architectures
  - Careful handling of tensor shapes and dimensions throughout the attention computation

- **Hybrid CNN-LSTM Model**:
  - Combined CNN and LSTM branches for parallel feature extraction
  - CNN branch: Two convolutional layers with pooling and dropout
  - LSTM branch: Two stacked LSTM layers with dropout
  - Branches combined through concatenation and processed by dense layers
  - This architecture leverages both spatial (CNN) and sequential (LSTM) patterns in the data
  - Designed to capture complementary information from different aspects of the spectral data

All models were implemented in PyTorch with careful attention to tensor dimensionality and efficient computation. The model implementations included proper handling of:
- Device management (CPU/GPU)
- Weight initialization
- Gradient calculation and backpropagation
- Batch processing
- NaN value prevention and handling
- Input/output tensor reshaping

Each model type was evaluated independently and optimized for the spectral data characteristics, with hyperparameters tuned specifically for the vomitoxin prediction task.

### 4.2 Training Methodology

The training process incorporated several advanced techniques to ensure model stability and optimal performance:

- **Batch Processing**: Models were trained with a batch size of 32, balancing computational efficiency and gradient stability

- **Early Stopping**: Custom implementation of early stopping with a patience of 10 epochs and best weight restoration to prevent overfitting

- **Gradient Clipping**: Applied with a maximum norm of 1.0 to prevent exploding gradients, which is particularly important for LSTM networks

- **Custom Loss Function**: Mean Squared Error (MSE) loss was used for optimization, aligned with the regression nature of the problem

- **Optimizer Selection**: Adam optimizer was used for all models due to its adaptive learning rate properties and good convergence characteristics

- **Training Monitoring**: Comprehensive tracking of both training and validation loss/MAE throughout the training process

- **Cross-Validation**: 5-fold cross-validation was implemented for hyperparameter tuning, ensuring robust parameter selection

### 4.3 Hyperparameter Optimization

A sophisticated Bayesian optimization approach was implemented to tune hyperparameters:

- **Gaussian Process Regression**: Used for modeling the hyperparameter-performance relationship with a Matern kernel

- **Acquisition Function**: Expected Improvement (EI) was used to balance exploration and exploitation during optimization

- **Parameter Ranges**:
  - LSTM: Units (32-256), dense layers (16-128), dropout (0.1-0.5), learning rate (1e-4 to 1e-2)
  - CNN: Filters (16-128), kernel size (3-7), dense layers (32-128), dropout (0.1-0.5), learning rate (1e-4 to 1e-3)
  - NN: Hidden layers (32-256), number of layers (1-4), dropout (0.1-0.5), learning rate (1e-4 to 1e-3)
  - Attention & Hybrid: Similar ranges adapted to their specific architectures

- **Optimization Process**:
  - 20 trials per model for hyperparameter optimization
  - 5-fold cross-validation for each parameter combination
  - Negative MSE as the optimization objective
  - Careful handling of potential NaN values during optimization

### 4.4 Evaluation Metrics

Models were evaluated using a comprehensive set of metrics to provide a thorough performance assessment:

- **Mean Absolute Error (MAE)**: Primary metric for assessing average prediction error magnitude
  - Provided an intuitive measure of prediction accuracy in log-transformed units
  - Less sensitive to outliers than MSE/RMSE

- **Root Mean Squared Error (RMSE)**: Secondary metric that penalizes larger errors more heavily
  - Important for applications where large prediction errors could have significant consequences
  - Used to compare model performance across different architectures

- **R² Score**: Measure of the proportion of variance explained by the model
  - Values closer to 1 indicate better predictive performance
  - Provides a standardized measure for comparison across different target scales

- **Learning Curves**: Training and validation loss/MAE plotted against epochs
  - Used to diagnose potential overfitting or underfitting
  - Helped assess model convergence characteristics

- **Actual vs Predicted Plots**: Visualization of model predictions against actual values
  - Included a perfect prediction line (y=x) for reference
  - Helped identify regions of the target range where the model performed better or worse

## 5. Key Findings and Suggestions for Improvement

### 5.1 Key Findings

1. **Feature Engineering Impact**: The derivative spectra, band ratios, and normalized difference indices significantly improved model performance beyond using raw spectral data alone. This highlights the importance of domain-specific feature engineering in hyperspectral data analysis.

2. **Dimensionality Reduction Insights**: PCA revealed that most of the variance (95%) could be captured in just 10 principal components, demonstrating the high redundancy in the original spectral data. This finding has significant implications for efficient data storage and processing of hyperspectral data in practical applications.

3. **Model Architecture Effectiveness**: The neural network models, particularly the LSTM and attention-based architectures, showed superior performance in capturing the complex relationships between spectral patterns and vomitoxin concentration. The hybrid CNN-LSTM model demonstrated the value of combining different architectural approaches to leverage complementary patterns in the data.

4. **Hyperparameter Sensitivity**: Bayesian optimization revealed high sensitivity to learning rate and network architecture parameters, while dropout rates showed more stability across a range of values. This suggests that careful tuning of learning rates is particularly important for neural network performance in this application.

5. **Target Transformation Benefit**: The log transformation of vomitoxin concentration values significantly improved model performance across all architectures. This confirms the importance of addressing the distributional characteristics of target variables in regression problems, particularly when dealing with biological concentration data that often follows log-normal distributions.

6. **Cross-validation Stability**: The 5-fold cross-validation results showed good stability across folds, indicating robust model performance that should generalize well to new data. This is particularly important given the limited sample size and the high dimensionality of the original feature space.

7. **Spectral Region Importance**: Feature importance analysis identified specific spectral regions that are strongly associated with vomitoxin concentration. These regions likely correspond to molecular absorption features related to the mycotoxin or associated fungal presence, providing potential targets for simplified sensor development.

8. **Attention Mechanism Effectiveness**: The attention model provided not only strong predictive performance but also valuable insights into which spectral regions were most relevant for prediction. The attention weights can be visualized to understand which parts of the spectrum the model is focusing on for its predictions.

9. **Robustness to Outliers**: The robust scaling approach combined with gradient clipping during training proved effective at maintaining model stability in the presence of outliers, which are common in spectral data due to sensor noise and atmospheric effects.

10. **NaN Handling Importance**: The implementations included careful handling of potential NaN values in both training and inference phases, which is critical for real-world applications where sensor data might have missing or corrupted values.

### 5.2 Suggestions for Improvement

1. **Advanced Feature Engineering**: Explore more sophisticated feature engineering techniques specific to spectral data, such as:
   - Savitzky-Golay filtering for noise reduction while preserving spectral features
   - Continuum removal to normalize spectra and enhance absorption features
   - Wavelet transforms for multi-scale analysis of spectral patterns
   - Integration of first and second derivatives for more comprehensive spectral characterization

2. **Transfer Learning**: Investigate the use of pre-trained neural networks on similar hyperspectral data, which could improve performance when working with limited sample sizes. This could involve adapting models trained on larger spectral datasets and fine-tuning them for vomitoxin prediction.

3. **Ensemble Methods**: Develop ensemble approaches combining predictions from multiple model architectures to leverage their complementary strengths. This could include:
   - Stacking different model types (CNN, LSTM, Attention)
   - Bagging approaches to reduce variance in predictions
   - Boosting techniques to focus on difficult-to-predict samples

4. **Explainable AI Extensions**: Enhance model interpretability through:
   - SHAP (SHapley Additive exPlanations) values for feature importance
   - Local interpretable model-agnostic explanations (LIME)
   - Attention visualization techniques for the attention-based models
   - Feature track-back to physical spectral bands to aid scientific understanding

5. **Data Augmentation**: Implement spectral data augmentation techniques to expand the training dataset:
   - Adding calibrated noise to simulate sensor variations
   - Spectral mixing with different proportions
   - Shifting and scaling spectra to simulate different illumination conditions
   - Synthetic sample generation using advanced techniques like GANs

6. **Real-time Processing Optimization**: Optimize models for deployment in field conditions:
   - Model quantization for reduced memory footprint
   - Architecture simplification for faster inference
   - Feature subset selection to focus only on the most predictive wavelengths
   - Development of progressive loading techniques for varying computational environments

7. **Integration with Additional Data Types**: Incorporate complementary data sources:
   - Weather and climate data that might affect fungal growth conditions
   - Temporal information about sample collection and storage
   - Field location and soil condition data
   - RGB imagery that might show visible symptoms of fungal infection

8. **Advanced Cross-validation Strategies**: Implement more sophisticated validation approaches:
   - Stratified k-fold cross-validation based on vomitoxin concentration levels
   - Leave-one-group-out cross-validation for spatially or temporally correlated samples
   - Nested cross-validation for more reliable hyperparameter tuning

9. **Uncertainty Estimation**: Add prediction uncertainty quantification:
   - Monte Carlo dropout for uncertainty estimation
   - Quantile regression techniques to predict confidence intervals
   - Bayesian neural networks for probabilistic predictions
   - Prediction intervals to accompany point estimates of vomitoxin concentration

10. **Pipeline Optimization**: Improve the end-to-end data processing pipeline:
    - Parallel processing for feature engineering steps
    - Caching of intermediate results for faster experimentation
    - Automated feature selection within the pipeline
    - Development of online learning capabilities for continuous model improvement

These enhancements would build upon the strong foundation established in the current work, further improving both prediction accuracy and practical utility of the hyperspectral vomitoxin prediction system.

## 6. Conclusion

This project demonstrated the effectiveness of machine learning approaches for predicting vomitoxin concentration from hyperspectral data. The comprehensive preprocessing pipeline, feature engineering techniques, and advanced modeling approaches resulted in predictive models with good accuracy.

The findings suggest that hyperspectral imaging, combined with appropriate data analysis techniques, has significant potential as a non-destructive method for vomitoxin detection in agricultural products. This approach could provide valuable tools for early detection and management of Fusarium infections in crops, ultimately contributing to food safety and reduced economic losses in agriculture.

The most successful models in this study were the LSTM and attention-based neural networks, indicating the importance of capturing sequential patterns and focusing on the most relevant spectral regions. The hybrid CNN-LSTM model demonstrated the value of combining multiple architectural approaches to leverage different aspects of the spectral data.

Key achievements of this project include:

1. Development of a comprehensive preprocessing pipeline specifically tailored for hyperspectral data
2. Creation of advanced feature engineering techniques that significantly improved model performance
3. Implementation of dimensionality reduction approaches that maintained information while drastically reducing data complexity
4. Development and optimization of multiple neural network architectures tailored to spectral data analysis
5. Identification of key spectral regions associated with vomitoxin concentration
6. Creation of a robust evaluation framework for comparing model performance
7. Implementation of sophisticated hyperparameter optimization techniques for model tuning

Future work should focus on:

1. Expanding the dataset with more samples across diverse growing conditions and vomitoxin concentration levels
2. Implementing the suggested improvements in feature engineering and model architecture
3. Developing practical deployment solutions for field applications
4. Integrating the models with IoT systems for real-time monitoring
5. Exploring transfer learning approaches to leverage knowledge from related spectral analysis tasks
6. Investigating multi-task learning to simultaneously predict multiple mycotoxin concentrations
7. Developing interpretability tools to provide actionable insights to agricultural practitioners

The comprehensive evaluation framework developed in this project provides a solid foundation for continuing research and development in this important area of precision agriculture. The code structure, documentation, and modular approach ensure that this work can be extended and adapted for related agricultural applications and similar hyperspectral analysis tasks. 