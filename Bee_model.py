#!/usr/bin/env python
# coding: utf-8

# In[285]:


# Loading necessary libraries and resources

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# In[286]:


# Reading in bee community data - describes bee presence for each site
df = pd.read_csv("./Community_Data.csv")


# In[287]:


# Checking data looks ok
df.head()


# In[288]:


# Counting total number of species present and adding it to the data frame (column = total_species)
# We do this by isolating all the columns containing species counts (so all but the first), then we count the number of these columns which contain a number greater than 0
species_cols = df.columns[1:]
df["total_species"] = (df[species_cols] > 0).sum(axis=1)


# In[289]:


# Counting total number of bees present and adding it to the data frame (column = total_bees)
# We do this by adding up all the bee counts across the species columns
df["total_bees"] = df[species_cols].sum(axis=1)


# In[290]:


# Calculating simpson's diversity index for each site and adding it to the data frame (column = diverity)

# Calculate proportions (n/N for each species, n = total number of focal species, N = total number of bees overall)
proportions = df[species_cols].div(df["total_bees"], axis=0)

# Simpson's Diversity Index (1 - sum((n/N)^2))
df["diversity"] = 1 - (proportions ** 2).sum(axis=1)


# In[291]:


# Exploring this diversity data
# Looking at statistical details (like min, max), and checking if there are any NA values
print(df["diversity"].describe())
print("\nNumber of NAs:", df["diversity"].isna().sum())


# In[292]:


# Histogram of diversity data
df["diversity"].hist()
# Very non-normal data, heavily left-skewed


# In[293]:


# Histogram of total number of species data
df["total_species"].hist()
# More normal looking, but a bit right skewed


# In[294]:


# Method 1: Classifying sites as 'high' or 'low' diversity
# High diversity = simpsons diversity index higher than the median
# Low diversity = simpsons diversitt index lower than the median
# Creating a column to store this classification (divseristy_class_simpsons)

simpson_median = df["diversity"].median()

df["diversity_class_simpsons"] = np.where(
    df["diversity"] > simpson_median,
    "high",
    "low"
)


# In[295]:


# Checking to see how many there are of each category (high diversity vs low diversity sites)
df["diversity_class_simpsons"].value_counts()


# In[296]:


# Method 2: Classifying sites as 'high' or 'low' diversity
# High diversity = number of species higher than the median
# Low diversity = number of species lower than the median
# Creating a column to store this classification (divseristy_class_species)

species_median = df["total_species"].median()
print(species_median)


# In[297]:


df["diversity_class_species"] = np.where(
    df["total_species"] > species_median,
    "high",
    "low"
)


# In[298]:


# Checking to see how many there are of each category (high diversity vs low diversity sites)
df["diversity_class_species"].value_counts()


# In[299]:


# Reading in site features data (things like latitude, longitude, water cover, forest cover, etc.)
site_data = pd.read_csv("Site_Data.csv")


# In[300]:


# Check it looks ok
site_data.head()


# In[301]:


# Creating a data table containing the bee community columns I want to keep
bee_summary = df[
    [
        "Site",
        "total_species",
        "total_bees",
        "diversity",
        "diversity_class_simpsons",
        "diversity_class_species"
    ]
]


# In[302]:


# Check it looks ok
bee_summary.head()


# In[303]:


# Now merging the bee_summary data frame and the site_data data frame
# This way, for each site, we have various measures of bee diversity and various site descriptors (e.g. water and forest cover)
# We will then be able to look at associations between bee diversity and site characteristics

bees = site_data.merge(
    bee_summary,
    on="Site",
    how="left"
)


# In[304]:


# Checking the merged table looks ok
bees.head()


# In[305]:


# I will remove columns that we won't be using in the model
# I will remove State and County as they are categorical variables which I don't wish to include in the model
# i will remove Northing and Eassting as we already have longitude and latitude
bees = bees.drop(columns=["State", "County", "Northing", "Easting"])


# In[306]:


# Looking at the correlations between all the variables

# Select only numeric columns
numeric_cols = bees.select_dtypes(include="number")

# Compute correlation matrix
corr = numeric_cols.corr()

# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Pairwise Correlation Between Numeric Variables")
plt.show()


# In[307]:


# Multi-variable scatter matrix to further look at relationships between variables

from pandas.plotting import scatter_matrix

a = scatter_matrix(bees, figsize=(16, 16))


# In[308]:


# We have several measures of bee diversity
# For the purposes of this model, I will use total number of bee species as the metric (rather than simpsons diversity index)
# The model will try to predict if a site has high or low diversity
# High diversity = more species than the median across sites
# Low diversoty = fewer species than the median across sites

# The site data looks at the ground cover of habitats like wetland and forest in the area surrounding a transect (within 200m, 1000m)
# Habitat cover at 200m and 1000m is strongly correlated for each site (makes sense)
# We will exclude 200m cover and just use 1000m cover

# Creating new data table with selected columns for model training:

# Create a list of columns for modeling
model_cols = [
    "Latitude",
    "Longitude",
    "1000m_open",
    "1000m_developed",
    "1000m_forest",
    "1000m_earlysuccessional",
    "1000m_wetlands",
    "diversity_class_species"  # This is the target variable
]

# Create new DataFrame for modeling
model_data = bees[model_cols].copy()


# In[309]:


# Checking new data frame looks ok
model_data.head()


# In[311]:


# Looking at colinearity again in this reduced dataframe

# Select only numeric columns
numeric_cols2 = model_data.select_dtypes(include="number")

# Compute correlation matrix
corr2 = numeric_cols2.corr()

# Plot a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr2, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Pairwise Correlations")
plt.show()

# For now I will not be removing any variables based on possible colinearity, as I'd like to run a model including everything first


# In[312]:


# Converting the diversity class (high vs low) into a numeric variable
# Convert 'high' → 1, 'low' → 0
model_data["diversity_class_species"] = model_data["diversity_class_species"].map({"low": 0, "high": 1})


# In[313]:


# Checking this looks ok
model_data.head()


# In[ ]:


# -----------------------------
# Logistic regression model
# -----------------------------


# In[314]:


# Preparing the data and training the logistic regression model

# Splitting data into features (X) and the data we are trying to predict (y)
X = model_data[['Latitude', 'Longitude', '1000m_open', '1000m_developed', '1000m_forest', '1000m_earlysuccessional', '1000m_wetlands']]
y = model_data['diversity_class_species']

# Standardize features (allow different ranges of numbers to be compared fairly)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
print("Prediction results of the first 5 test samples：", y_pred[:5])
print("Top 5 real results：", y_test.values[:5])


# In[315]:


# Checking how well the model did

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy：{accuracy:.2f}（Proportion of correct predictions）")

# Draw confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['low diversity', 'high diversity'])
disp.plot(cmap='Blues')
plt.title('matrix')
plt.show()

# See which features are important
feature_importance = pd.DataFrame({
    'features': ['(Latitude)', '(Longitude)', '(1000m_open)', '(1000m_developed)', '(1000m_forest)', '(1000m_earlysuccessional)', '(1000m_wetlands)'],
    'weights': model.coef_[0]
})
print("importance of features：")
print(feature_importance)


# In[316]:


# Summary of confusion matrix:
# Top left: true negatives: 8
# Top right: flase positives (site has low diversity, but model predicts high): 6
# Bottom left: false negatives (site has high diversity, but model predicts low): 6
# Bottom right: true positives: 10


# In[317]:


# Our model currently has an accuracy of 0.60 (not great)
# I will try some tweaks to see if I can improve the model at all


# In[321]:


# We will try recursive feature elimination (removing features which do not improve the model's accuracy)
from sklearn.feature_selection import RFE

# Rank all features by their importance
rfe = RFE(estimator=model, n_features_to_select=1)
rfe.fit(X_train, y_train)

# Feature names
feature_names = ['Latitude', 'Longitude', '1000m_open', '1000m_developed', '1000m_forest', 
                 '1000m_earlysuccessional', '1000m_wetlands']

# Rank features using RFE (1 = most important)
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=1)  # rank all features
rfe.fit(X_train, y_train)

# Get feature rankings
ranking = rfe.ranking_
feature_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Ranking': ranking
}).sort_values('Ranking')
print("Feature rankings (1 = most important):")
print(feature_ranking)


# In[323]:


# Iteratively removing lowest-ranked features and storing model accuracy

# Sort features from lowest importance (highest rank) to highest
features_sorted_by_rank = feature_ranking.sort_values('Ranking', ascending=False)['Feature'].tolist()

results = []

# Start with all features and iteratively remove one at a time
for i in range(len(features_sorted_by_rank)):
    selected_features = features_sorted_by_rank[i:]  # remove lowest i features
    indices = [feature_names.index(f) for f in selected_features]  # get column indices
    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices]

    # Train logistic regression
    model_iter = LogisticRegression(max_iter=1000)
    model_iter.fit(X_train_selected, y_train)

    # Compute test accuracy
    accuracy = model_iter.score(X_test_selected, y_test)

    results.append({
        'Num_features': len(selected_features),
        'Features': selected_features,
        'Test_accuracy': accuracy
    })

# Convert results to a data frame
results_df = pd.DataFrame(results)
print("\nModel performance as features are removed:")
print(results_df)

# It appears that removing features will not improve the model's accuracy


# In[324]:


# -----------------------------
# Random forest model
# -----------------------------


# In[325]:


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)


# In[326]:


# Predict on test set
y_pred = rf_model.predict(X_test)
print("Prediction results of the first 5 test samples:", y_pred[:5])
print("Top 5 real results:", y_test.values[:5])


# In[327]:


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest test accuracy: {accuracy:.2f}")


# In[328]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['low diversity', 'high diversity'])
disp.plot(cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()


# In[338]:


# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("Random Forest feature importance:")
print(feature_importance)


# In[ ]:


# -----------------------------
# K means clustering model
# -----------------------------


# In[341]:


from sklearn.cluster import KMeans

# Determine number of clusters

# We will try 2 using clusters (hopefully they'll be close to high vs low diversity)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Cluster assignments
clusters = kmeans.labels_
print("Cluster assignments for first 10 sites:", clusters[:10])

# Add cluster labels to original DataFrame
clustered_data = model_data.copy()
clustered_data['Cluster'] = clusters


# In[343]:


# Compare clusters to actual diversity class

# Confusion matrix between clusters and actual classes
cm = confusion_matrix(clustered_data['diversity_class_species'], clustered_data['Cluster'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap='Blues')
plt.title('K-Means Clusters vs Actual Diversity Class')
plt.show()


# In[346]:


from sklearn.metrics import adjusted_rand_score

# Adjusted Rand Index (measures similarity between clusters and true labels, 1 = perfect match)
ari = adjusted_rand_score(clustered_data['diversity_class_species'], clustered_data['Cluster'])
print(f"Adjusted Rand Index between clusters and true labels: {ari:.2f}")


# In[347]:


# Cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
print("Cluster centers (original scale):")
print(cluster_centers)

