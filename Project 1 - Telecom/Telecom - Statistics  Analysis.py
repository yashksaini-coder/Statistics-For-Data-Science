import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Telecom Dataset.csv")
print(df.shape)
print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.columns)
def count_values(dataframe, column_list):
    for column in column_list:
        print(f"Value counts for {column}:")
        print(dataframe[column].value_counts())
        print("-" * 20)

l = ["Blue", "Wi_Fi", "Tch_Scr", "Ext_Mem"]
print(count_values(df,l)) 


### The children want phones that have the following: Bluetooth, WiFi, touch screen and external memory support
### Create a logical condition for this situation and store the logical values as "con1"
con1 = (df['Blue'] == 'yes') & (df['Wi_Fi'] == 'yes') & (df['Tch_Scr'] == 'yes') & (df['Ext_Mem'] == 'yes')
print(con1.head())

l2= ['Px_h','Px_w']
print(count_values(df,l2))

df['Px'] = df['Px_h'] + df['Px_w']
print(df['Px'].value_counts)()

# Create a histogram of the "Px" feature and also show the mean and the median
# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Px'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Px Feature')
plt.xlabel('Total Resolution of the Screen (Px)')
plt.ylabel('Frequency')
# Adding mean and median lines
mean_value = df['Px'].mean()
median_value = df['Px'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()


### The children want phones that have good screen resolutions
### Consider the phones that have screen resolutions greater than or equal to the median value in the data set
### Create a logical condition for this situation and store the logical values as "con2"

# Calculate the median screen resolution
median_resolution = df['Px'].median()
con2 = df['Px'] >= median_resolution
print(con2.head())

l3 = ['Scr_h','Scr_w']
print(count_values(df,l3))

# Create a new feature called "Scr_d" which stores the length of the diagonal of the screen of the phone
df['Scr_d'] = np.sqrt(df['Scr_h']**2 + df['Scr_w']**2)
df['Scr_d'].head()

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Scr_d'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Scr_d Feature')
plt.xlabel('Length of Diagonal of the Screen (Scr_d)')
plt.ylabel('Frequency')
# Adding quartile lines
q25 = df['Scr_d'].quantile(0.25)
q50 = df['Scr_d'].quantile(0.50)
q75 = df['Scr_d'].quantile(0.75)
plt.axvline(q25, color='red', linestyle='dashed', linewidth=2, label=f'Q1: {q25:.2f}')
plt.axvline(q50, color='green', linestyle='dashed', linewidth=2, label=f'Median: {q50:.2f}')
plt.axvline(q75, color='blue', linestyle='dashed', linewidth=2, label=f'Q3: {q75:.2f}')
plt.legend()
plt.show()


### The children want phones that have very good screen sizes
### Consider the phones that have screen sizes greater than or equal to the upper quartile value in the data set
### Create a logical condition for this situation and store the logical values as "con3"

# Calculate the upper quartile value for the Scr_d feature
upper_quartile = df['Scr_d'].quantile(0.75)
# Create the logical condition con3
con3 = df['Scr_d'] >= upper_quartile

print(con3.head())

# Let's tackle these features: "PC", "FC"
l4 = ['PC','FC']
print(count_values(df,l4))

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['PC'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of PC Feature')
plt.xlabel('Primary Camera Resolution (PC)')
plt.ylabel('Frequency')
# Adding mean and median lines
mean_value = df['PC'].mean()
median_value = df['PC'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()


# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['FC'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of FC Feature')
plt.xlabel('Front camera resolution (in MP) (FC)')
plt.ylabel('Frequency')
# Adding mean and median lines
mean_value = df['FC'].mean()
median_value = df['FC'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()


# Calculate the mean values for PC and FC features
mean_pc = df['PC'].mean()
mean_fc = df['FC'].mean()
# Create the logical condition con4
con4 = (df['PC'] >= mean_pc) & (df['FC'] >= mean_fc)
print(con4.head())

# Let's tackle these features: "Int_Mem", "Bty_Pwr", "RAM"
l5 = ["Int_Mem", "Bty_Pwr", "RAM"]
print(count_values(df,l5))

# Plotting the histogram
plt.figure(figsize=[12,6])
plt.hist(df['Int_Mem'], bins=20,color='skyblue',edgecolor='black')
plt.title('Histogram of Int_Mem feature')
plt.xlabel('Internal Memory (Int Mem)')
plt.ylabel('Frequency')
mean_value = df['Int_Mem'].mean()
median_value = df['Int_Mem'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Bty_Pwr'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Bty_Pwr Feature')
plt.xlabel('Battery Power (Bty_Pwr)')
plt.ylabel('Frequency')
# Adding mean and median lines
mean_value = df['Bty_Pwr'].mean()
median_value = df['Bty_Pwr'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['RAM'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of RAM Feature')
plt.xlabel('Random Access Memory (RAM)')
plt.ylabel('Frequency')

# Adding mean and median lines
mean_value = df['RAM'].mean()
median_value = df['RAM'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()

# Calculate the mean values for Int_Mem, Bty_Pwr, and RAM features
mean_int_mem = df['Int_Mem'].mean()
mean_bty_pwr = df['Bty_Pwr'].mean()
mean_ram = df['RAM'].mean()

# Create the logical condition con5
con5 = (df['Int_Mem'] >= mean_int_mem) & (df['Bty_Pwr'] >= mean_bty_pwr) & (df['RAM'] >= mean_ram)
print(con5.head())

# Let's tackle these features: "Depth", "Weight"
l6 = ["Depth", "Weight"]
print(count_values(df,l6))

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Depth'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Depth Feature')
plt.xlabel('Depth of the Mobile Phone (Depth)')
plt.ylabel('Frequency')
# Adding mean and median lines
mean_value = df['Depth'].mean()
median_value = df['Depth'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Weight'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Weight Feature')
plt.xlabel('Weight of the Mobile Phone (Weight)')
plt.ylabel('Frequency')
# Adding mean and median lines
mean_value = df['Weight'].mean()
median_value = df['Weight'].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.legend()
plt.show()

# Calculate the median values for Depth and Weight features
median_depth = df['Depth'].median()
median_weight = df['Weight'].median()
# Create the logical condition con6
con6 = (df['Depth'] <= median_depth) & (df['Weight'] <= median_weight)
print(con6.head())

# Subset the DataFrame using logical conditions
df1 = df[con1 & con2 & con3 & con4 & con5 & con6]
print(df1.head())



# In[48]:


# Get the dimensions of the dataframe
df1.shape


# In[49]:


# Sort the dataframe according to the "Price" feature in ascending order and display it
df1_sorted = df1.sort_values(by='Price',ascending=True)


# In[50]:


df1_sorted.head()


# Observations:
# 
# Based on all the logical conditions obtained through analysis of the features, we are left with three phones.
# 
# The most expensive of these phones is the "TYS938L" model and the least expensive is the "TVF078Y" model.
# 
# We could let the children choose from these three phones as per their preferences.

# # Task 9 - Study the variability of the features in the original data set

# ### Calculate the ratio of the standard deviation to the mean for all the numerical features in the dataframe
# ### Store these values in a new series wherein the rows are the features and the only column is the calculated ratio
# 

# In[51]:


# Calculate the ratio of standard deviation to mean for all numerical features
deviations = (df1_sorted.select_dtypes(include='number').std() / df1_sorted.select_dtypes(include='number').mean()).rename("Ratio")


# In[52]:


deviations.head()


# In[53]:


# Sort the "deviations" Series in descending order
deviations_sorted = deviations.sort_values(ascending=False)


# In[54]:


deviations_sorted.head()
