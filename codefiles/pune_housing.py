# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Pune_House_Data.csv")
df.head()

df.columns
df.shape
df.info()
df.describe()

miss_percentage = (df.isnull().sum())/(len(df))*100
print(miss_percentage)
# 41% missing data in society column

#%%

#looking at square feet column
df['total_sqft'].values
#convert object datatype to int
df['total_sqft'] =pd.to_numeric(df['total_sqft'],
   errors='coerce')

# check 
df.info()

#%%

# looking at size column
df['size'].value_counts()

df['size'].unique()

df['size'].nunique()

# there needs to be uniformity in this series values
# if the value contains bedroom change to BHK
# if it contains Rk keep it as it is, 13 values with RK 

# we need to standardize the size column

# Standardize the 'size' column
def clean_size(value):
    if pd.isna(value):
        return value  # Keep NaN as is
    value = value.strip()
    if "Bedroom" in value:
        return value.replace("Bedroom", "BHK")
    elif "RK" in value:
        return value  # Keep "RK" as is
    return value  # Keep other values as is

df['size'] = df['size'].apply(clean_size)

# top 5 variants in flats in terms of sizes
df['size'].value_counts(ascending=False)[:5]

# 2 BHK flats are the highest at 5528

twobhk_concentration = 5528/(df.shape[0])*100
print(twobhk_concentration)
# 42% flats are 2bhk

# looking at 5 bhk concentration
fivebhk_concentration = 356/(df.shape[0])*100
print(fivebhk_concentration)
# 2.6% flats are 5bhk - technically they are outliers here

# we can create a seperate column which contains ranges
# 3BHk and above
# 2bhk and below

# Define a function to classify the size into "3BHK and above" or "2BHK and below"
def classify_flat_variant(size):
    if pd.isna(size):
        return None  # Handle missing values by returning None
    size = size.strip()
    if 'BHK' in size:
        # if 3BHK then split as ["3","BHK"]
        bhk = int(size.split(' ')[0]) 
        # Extract the 1st element in the split
        if bhk >= 3:
            return '3BHK and above'
        else:
            return '2BHK and below'
    elif 'RK' in size:
        return '2BHK and below'  # Treat rooms (RK) as below 2BHK
    else:
        return None  # In case there's an unexpected format

# Apply the function to the 'size' column and create a new column 'Flat variant'
df['Flat variant'] = df['size'].apply(classify_flat_variant)

df.columns

# new dataset exported for checking the column creation
df.to_csv("df_flatvariant.csv")


#%%
df['area_type'].value_counts()

# Standardize 'area_type' column by replacing spaces with underscores
df['area_type'] = df['area_type'].str.replace(' ', '_')

# Check the unique values after cleaning
print(df['area_type'].unique())

# against the built up area looking at sqft
df.groupby('area_type')['total_sqft'].agg(['min','max','mean'])

#%%
df.loc[df['availability']=="Ready To Move"].shape
# 10581 records are in Ready to Move status

df.loc[df['availability']=="Immediate Possession"].shape
# 16 records are in Immediate Possession status

# we can create 2 seperate columns for days and date




#%%

df['bath'] = df['bath'].fillna(0)  # Replace NaN with 0 or a default value
df['balcony'] = df['balcony'].fillna(0)

# Create a count plot for 'bath'
plt.figure(figsize=(8, 5))
sns.countplot(x='bath', data=df, palette='viridis')
plt.title('Count Plot of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')
plt.show()

# Create a count plot for 'balcony'
plt.figure(figsize=(8, 5))
sns.countplot(x='balcony', data=df, palette='magma')
plt.title('Count Plot of Balconies')
plt.xlabel('Number of Balconies')
plt.ylabel('Count')
plt.show()


# Create a histogram for both 'bath' and 'balcony'
plt.figure(figsize=(8, 5))
sns.histplot(df['bath'],
             kde=True, 
             color='blue',
             label='Bathrooms', bins=10)

sns.histplot(df['balcony'], 
             kde=True, 
             color='orange',
             label='Balconies', bins=10)

plt.title('Histogram of Bathrooms and Balconies')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#%%

# looking at flat variants

df.groupby('Flat variant')['price'].agg(['min','max','mean'])

df['Flat variant'].value_counts()

df['Flat variant'].str.replace(" ","_") #improves readability


#%%


# Plotting price distribution by area_type
plt.figure(figsize=(20, 10))
sns.boxplot(x='area_type', y='price', data=df)
plt.xticks(rotation=90)  # Rotate labels for better readability
plt.title('Price Distribution by Location')
plt.xlabel('Location')
plt.ylabel('Price in Lakhs')
plt.show()

# we can check there are outliers in all the area types
# take price column 

# Selecting only numerical columns for outlier analysis

numerical_cols = df.select_dtypes(include=['float64', 'int64'])
print(numerical_cols)

# Function to calculate IQR and count outliers in each numerical column
def count_iqr_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    lower_bound = Q1 - 1.5 * IQR
    print(lower_bound)
    upper_bound = Q3 + 1.5 * IQR
    print(upper_bound)
    
    # Counting the outliers before applying treatment
    outliers_count = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)].shape[0]
    return outliers_count

# Applying the function to each numerical column and storing the results
outlier_counts = {col: count_iqr_outliers(df, col) for col in numerical_cols.columns}

# Outputting the results
for col, count in outlier_counts.items():
    print(f"Number of outliers in {col}: {count}")

# We need to perform outlier treatment here
# We prefer Winsorization/Capping for data rentention purposes
# We will now prefer capping lower bound and upper bound values

# Cap outliers at the calculated lower and upper bounds based on IQR
for col in numerical_cols.columns:
    Q1 = df[col].quantile(0.25)
    print(Q1)
    Q3 = df[col].quantile(0.75)
    print(Q3)
    IQR = Q3 - Q1
    print(IQR)
    lower_bound = Q1 - 1.5 * IQR
    print(lower_bound)
    upper_bound = Q3 + 1.5 * IQR
    print(upper_bound)

    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)  # Capping/Winsorizing



# Verify capping (optional - check if any values are still outside the bounds)
for col in numerical_cols:
    values_below_lower = (df[col] < lower_bound).sum()
    values_above_upper = (df[col] > upper_bound).sum()
    print(f"Values below lower bound in {col} after capping: {values_below_lower}") # Should be 0
    print(f"Values above upper bound in {col} after capping: {values_above_upper}") # Should be 0

#%%

# society column seems irrelevant we will drop it

df.drop(columns='society',axis=1,inplace=True)

df.isnull().sum()

# in total_sqft column there are 247 null values we will impute them 
# check skewness
skewness = df['total_sqft'].skew()
print(f"Skewness of total_sqft: {skewness}")

# A skewness value close to 0 indicates a symmetrical distribution.
# A positive skewness value indicates a distribution with a long tail to the right.
# A negative skewness value indicates a distribution with a long tail to the left.


# For total_sqft , it is moderately right skewed 

# in case you wish to visualize the skewness

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['total_sqft'], kde=True)
plt.title('Histogram of Total Square Feet')
plt.xlabel('Total Square Feet')
plt.ylabel('Frequency')
plt.show()

# if moderately skewed median happens to be a good option

# calculate the median
median_total_sqft = df['total_sqft'].median()
print("Median of 'total_sqft':", median_total_sqft)

# impute total_sqft using median 
df['total_sqft'].fillna(median_total_sqft, inplace=True)

missing_after_imputation = df['total_sqft'].isnull().sum()
print("Missing values in 'total_sqft' after imputation:", missing_after_imputation)


#%%

# size and flat_variant column contains 16 Null values each
# they both are object datatypes cannot be imputed using mean,median


df.info()

# Check the most frequent category in 'Flat variant' and 'size'
most_frequent_flat_variant = df['Flat variant'].mode()[0]
most_frequent_size = df['size'].mode()[0]

# Impute missing values with the most frequent category
df['Flat variant'].fillna(most_frequent_flat_variant, inplace=True)
df['size'].fillna(most_frequent_size, inplace=True)

df.isnull().sum()

# all the missing values are now being imputed

# we can now export this dataset to a csv which contains no null
df.to_csv("NoNulldf.csv")


#%%

# now work on the availablity column
# check what can be done
# it contains different formats 
# check for unique categories and what can we do with Dates