# Predicting Bee Species Diversity from Agricultural Site Characteristics

---

## Project Overview

This project explores how habitat characteristics influence bee species diversity across multiple sampling sites.  
Bee diversity is an important indicator of ecosystem health, and understanding how landscape features relate to diversity can help inform conservation and land management decisions.

The analysis combines bee community data with site-level habitat information and applies both supervised and unsupervised machine learning approaches. The primary aim is to predict bee species richness at an agricultural site based on:
- Site latitude and longitude  
- Proportional cover of habitat types, including:
  - Open water  
  - Developed / urban land  
  - Forest  
  - Agricultural / early successional habitat  
  - Wetlands  

---

## Files

Bee_diversity_model.ipynb  
A Jupyter notebook containing all data processing, analysis, and modelling steps.

Bee_model.py  
A Python script version of the modelling workflow.

Community_Data.csv  
A dataset containing bee species abundance data for each site.

Site_Data.csv  
A dataset containing geographic and habitat characteristics for each site.

---

## Data Description

The data used in this project were originally sourced from the following publication:

Landsman, A. P., Simanonok, M. P., Savoy-Burke, G., Bowman, J. L., and Delaney, D. A.  
Geographic drivers more important than landscape composition in predicting bee beta diversity and community structure.  
Ecosphere 15, e4819 (2024).

Data were collected by citizen science volunteers using bee bowl transects across 99 forest sites in Maryland, Delaware, Washington, D.C., and northern Virginia, USA, during 2014.

### Community_Data.csv

This dataset includes:
- Site name (Site)
- A community matrix containing abundance counts for individual bee species collected at each site

### Site_Data.csv

This dataset includes:
- Site name (Site)
- Geographic coordinates in decimal degrees (Latitude, Longitude)
- State and county identifiers
- Geographic coordinates in meters north (Northing) and east (Easting) of UTM Zone 18 origin
- Proportional land cover of habitat types (open water, developed, forest, agriculture and early successional, wetlands) measured within 200 m and 1000 m buffers around each sampling transect

---

## Methods Summary

### Bee Diversity Metrics

Three measures of bee diversity were calculated for each site:

- **Total number of species**  
  The number of bee species present at a site.

- **Total number of bees**  
  The total number of individual bees observed across all species at a site.

- **Simpson’s Diversity Index**  
  Calculated as:  
  D = 1 − Σ(n / N)²  
  where n is the number of individuals of a single species and N is the total number of bees at the site.

Sites were classified as **high** or **low diversity** based on whether the number of species present exceeded the median species richness across all sites.

---

### Modelling Approaches

The following machine learning models were applied:

- Logistic regression
- Random forest classifier
- K-means clustering

Habitat variables measured within 1000 m buffers were used for modelling to reduce collinearity with 200 m habitat measures.

---

## How to Run the Analysis

### Requirements

Python version 3.8 or higher is required.

The following Python libraries are used:
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn

## Installation Instructions.

This project used uv, a lightweight Python package and environment manager.
If uv is not already installed, installation instructions are available in the uv documentation.

### Installing packages using uv

To install all required packages, run:

uv add pandas numpy matplotlib seaborn scikit-learn jupyterlab

To add an individual package to the environment, use:

uv add <package-name>

### Running the Analysis

1. Clone or download the repository
2. Ensure the CSV data files are located in the same directory as the notebook
3. Open the Jupyter notebook Bee_diversity_model.ipynb
4. Run all cells from top to bottom

---

## Key Results

- Logistic regression and random forest models achieved moderate predictive accuracy (approximately 60%)
- Habitat variables alone were insufficient to strongly predict high versus low bee diversity
- K-means clustering showed poor alignment with observed diversity classifications

These results suggest that bee diversity is influenced by complex ecological factors beyond coarse habitat cover metrics.

---

## Limitations

- Relatively small sample size
- Binary classification of an inherently continuous diversity signal
- Semi-correlated environmental predictors
- Habitat variables may not capture fine-scale floral or nesting resources important for bees

---

## Author

Miranda Johnstone
