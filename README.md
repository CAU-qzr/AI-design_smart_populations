# AI-design smart populations
Artificial intelligence designs smart maize populations for productive agriculture within planetary boundaries

# Dataset Description

## Raw Data Files

### 1. AIsp_origindata.xlsx
- **Purpose**: Raw dataset for building the AIsp_TabPFN model
- **Content**: Contains all original features and target variables used for AIsp model development
---"""
Soil: soil organic matter (SOM).
Climate: average daily maximum temperature (Tmax), average daily minimum temperatures (Tmin), cumulative precipitation (Prep), cumulative solar radiation (Radn), and standardized precipitation–evapotranspiration index (SPEI).
Management: Tillage and Irrigation.
Thermal indices: growing degree days (GDD) and extreme degree days (EDD).
Genotype: canopy architecture (Canopy). 
Target variable: optimal planting density (OPD).
Note: Optimal yield (Yopt): This variable was not utilized in the modeling process, but subsequent steps require mapping from OPD to Yopt by canopy (smart vs. non-smart).
---"""

### 2. AInm_origindata.xlsx
- **Purpose**: Raw dataset for building the AInm_CatBoost model
- **Content**: Contains all original features and target variables used for AInm model development
---"""
Soil: soil organic matter (SOM), soil total nitrogen (TotalN) and pH.
Climate: average daily maximum temperature (Tmax), average daily minimum temperatures (Tmin), cumulative precipitation (Prep), cumulative solar radiation (Radn), and standardized precipitation–evapotranspiration index (SPEI).
Management: Tillage and Irrigation.
Thermal indices: growing degree days (GDD) and extreme degree days (EDD).
Productivity: optimal yield (Yopt).
Target variable: optimal nitrogen application (Nopt).
---"""

# Python Code Files

## 1. AIsp_compare.py
- **Function**: Compares performance of different algorithms for AIsp model construction
- **Process**: 
  - Evaluates multiple machine learning algorithms
  - Identifies the optimal algorithm based on performance metrics
  - Outputs comparative analysis results

## 2. AIsp_TabPFN.py
- **Function**: Implements the final AIsp model using the selected optimal algorithm
- **Process**:
  - Performs 10-fold cross-validation with SHAP interpretation
  - Retrains the model using the entire dataset for final deployment
  - Generates model explanations and feature importance analysis

## 3. AInm_compare.py
- **Function**: Compares performance of different algorithms for AInm model construction  
- **Process**:
  - Evaluates multiple machine learning algorithms
  - Identifies the optimal algorithm based on performance metrics
  - Outputs comparative analysis results

## 4. AInm_TabPFN.py
- **Function**: Implements the final AInm model using the selected optimal algorithm
- **Process**:
  - Performs 10-fold cross-validation with SHAP interpretation
  - Retrains the model using the entire dataset for final deployment
  - Generates model explanations and feature importance analysis
