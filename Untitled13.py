#!/usr/bin/env python
# coding: utf-8

# In[38]:


pip install seaborn


# In[45]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Step 1: Load Multiple Datasets ==========
folder_path = r"C:\Users\beraj\OneDrive\Desktop\Cleaned_Datasets" 
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')] 

dfs = {}

for file in file_list:
    file_path = os.path.join(folder_path, file)
    dfs[file] = pd.read_csv(file_path)
    print(f"Loaded {file} with shape {dfs[file].shape}")

# ========== Step 2: Define Automated EDA Function ==========
def perform_eda(df, name):
    
    print(f"\n================= EDA for {name} =================")
    print("\nBasic Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicate_count}")
    if 'Year' in df.columns:
        num_cols = df.select_dtypes(include=['number']).columns
        num_cols = [col for col in num_cols if col != 'Year']

        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=df['Year'], y=df[col])
            plt.xticks(rotation=45)
            plt.title(f"{col} over Years in {name}")
            plt.xlabel("Year")
            plt.ylabel(col)
            plt.show()


# ========== Step 3: Run EDA for All Datasets ==========
for file_name, df in dfs.items():
    perform_eda(df, file_name)

# ========== Step 4: Generate Summary Report for All Datasets ==========
summary_list = []

for file_name, df in dfs.items():
    summary_list.append({
        "Dataset": file_name,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum()
    })

summary_df = pd.DataFrame(summary_list)
print("\nOverall Dataset Summary:")
print(summary_df)

# ========== Step 5: Save Cleaned Datasets ==========
output_folder = r"C:\Users\beraj\OneDrive\Desktop\Cleaned_Datasets"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name, df in dfs.items():
    df_cleaned = df.dropna()
    cleaned_file_path = os.path.join(output_folder, file_name)
    df_cleaned.to_csv(cleaned_file_path, index=False)
    print(f"Saved cleaned dataset: {file_name}")


# In[2]:


for file_name, df in dfs.items():
    print(f"Dataset:{file_name}")
    print(df.columns)
    print("=" * 50)


# In[3]:


df_gdp = dfs['GDP growth.csv'] 
print(df_gdp.columns) 


# In[4]:


from statsmodels.tsa.stattools import adfuller

# ADF test to check if data is stationary or not
df_gdp_growth = dfs['GDP growth.csv'] 
if 'Percentage_Growth' in df_gdp_growth.columns:
    result = adfuller(df_gdp_growth['Percentage_Growth'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
else:
    print("'Percentage_Growth' column not found in the dataset.")
    
    #p value not less that 0.05


# In[5]:


#Checking if data is stationary after difference
for file_name, df in dfs.items():

    if 'Percentage_Growth' in df.columns:  
        df['Percentage_Growth_diff'] = df['Percentage_Growth'] - df['Percentage_Growth'].shift(1)
        
        result = adfuller(df['Percentage_Growth_diff'].dropna())
        print(f"ADF Statistic for {file_name}: {result[0]}")
        print(f"p-value for {file_name}: {result[1]}")


# In[6]:


gdp_growth = dfs['GDP growth.csv'] 
cpi_data=dfs['CPI_redone.csv']
exports_data=dfs['Exports of goods and services_redone.csv']
gdp_deflator_data=dfs['GDP deflator_redone.csv']

df_all = gdp_growth.merge(cpi_data, on='Year').merge(exports_data, on='Year').merge(gdp_deflator_data, on='Year')

df_all.set_index('Year', inplace=True)
df_all.index = df_all.index.astype(int)  # Ensure it's an integer index


# In[7]:


from statsmodels.tsa.stattools import adfuller

# Run ADF test for each numerical column
for col in df_all.columns:
    if col != "Year" and df_all[col].dropna().nunique() > 1:  # 'Year' is not a time series value
        result = adfuller(df_all[col].dropna())  # Drop NaNs before ADF test
        print(f"\nADF Test for {col}:")
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")

        if result[1] < 0.05:
            print("✅ Stationary (p < 0.05)")
        else:
            print("❌ Not Stationary (p >= 0.05) - Consider differencing or transformation")


# In[8]:


df_all['Percentage_Growth'] = gdp_growth['Percentage_Growth_diff']


# In[9]:


from statsmodels.tsa.stattools import adfuller

# Run ADF test for each numerical column
for col in df_all.columns:
    if col != "Year" and df_all[col].dropna().nunique() > 1:  # 'Year' is not a time series value
        result = adfuller(df_all[col].dropna())  # Drop NaNs before ADF test
        print(f"\nADF Test for {col}:")
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")

        if result[1] < 0.05:
            print("✅ Stationary (p < 0.05)")
        else:
            print("❌ Not Stationary (p >= 0.05) - Consider differencing or transformation")


# In[10]:


import numpy as np
from statsmodels.tsa.stattools import adfuller

# Log transformation and differencing for Exports of Goods and Services
df_all['Exports_log'] = np.log(df_all['Exports of Goods and Services'])
df_all['Exports_log_diff'] = df_all['Exports_log'].diff()

# Run ADF test and update columns if stationary
for col in df_all.columns:
    if col != "Year" and col != "year" and df_all[col].dropna().nunique() > 1:  # Skip non-time series values
        result = adfuller(df_all[col].dropna())  # Drop NaNs before ADF test
        p_value = result[1]
        
        # If the series is stationary, update the dataframe column
        if p_value < 0.05:
            print(f"\nADF Test for {col}:")
            print(f"ADF Statistic: {result[0]}")
            print(f"p-value: {p_value}")
            print("✅ Stationary (p < 0.05)")

            # For specific columns (like Exports_log_diff), update the dataframe
            if col == 'Exports_log_diff':
                df_all['Exports of Goods and Services'] = df_all['Exports_log_diff']  # Update the original column
                print(f"Updated {col} in df_all")



# In[11]:


from statsmodels.tsa.stattools import adfuller

# Run ADF test for each numerical column
for col in df_all.columns:
    if col != "Year" and df_all[col].dropna().nunique() > 1:
        result = adfuller(df_all[col].dropna()) 
        print(f"\nADF Test for {col}:")
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")

        if result[1] < 0.05:
            print("✅ Stationary (p < 0.05)")
        else:
            print("❌ Not Stationary (p >= 0.05) - Consider differencing or transformation")


# In[12]:


print(df_all)


# In[46]:


print(df_all.columns)


# In[22]:


from statsmodels.tsa.stattools import adfuller

for col in df_all.columns:
    if col != "Year" and df_all[col].dropna().nunique() > 1:
        result = adfuller(df_all[col].dropna()) 
        print(f"\nADF Test for {col}:")
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")

        if result[1] < 0.05:
            print("✅ Stationary (p < 0.05)")
        else:
            print("❌ Not Stationary (p >= 0.05) - Consider differencing or transformation")


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

selected_features = ['GDP_In_Billion_USD', 'Per_Capita_in_USD', 'Percentage_Growth_diff',
                     'CPI', 'Exports of Goods and Services', 'GDP_deflator', 'Exports_log_diff']

missing_cols = [col for col in selected_features if col not in df_all.columns]
if missing_cols:
    print(f"Missing columns in df_all: {missing_cols}")
else:
    corr_df = df_all[selected_features].corr()

    print(corr_df)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap for Stationary Data (df_all)")
    plt.show()


# In[ ]:




