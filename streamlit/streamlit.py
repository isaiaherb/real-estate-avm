import streamlit as st 
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df_property = pd.read_csv('/Users/isaiaherb/Documents/Northbow/WholeFoods.csv')
file_path_main = '/Users/isaiaherb/Documents/Northbow/mrtssales92-present.xlsx'

def fetch_census_data(file_path, sheet):
    df = pd.read_excel(file_path, sheet_name=sheet, skiprows=4)
    df.set_index('Unnamed: 1', inplace=True)
    df = df.drop(columns='Unnamed: 0')
    df_filtered = df.iloc[[27, 28, 29]]
    df_transposed = df_filtered.transpose()
    df = df_transposed.rename_axis('Date', axis='index')
    df = df.rename_axis('Store Type', axis='columns')    
    return df

dfs = [fetch_census_data(file_path_main, f'{year}') for year in range(2010, 2023)]
df_retail_sales = pd.concat(dfs)

df_property['sale_date'] = pd.to_datetime(df_property['sale_date'], format='%m/%Y')
df_retail_sales.reset_index(inplace=True)
df_retail_sales.rename(columns={'index': 'Date'}, inplace=True)
formats = ['%b. %Y', '%b %Y']
for fmt in formats:
    try:
        df_retail_sales['Date'] = pd.to_datetime(df_retail_sales['Date'], format=fmt)
        break  
    except ValueError:
        pass 

df_retail_sales = df_retail_sales[df_retail_sales['Date'] != 'TOTAL']
df_retail_sales['Date'] = pd.to_datetime(df_retail_sales['Date'], format='mixed')
df_property['Month_Year'] = df_property['sale_date'].dt.to_period('M')
df_retail_sales['Month_Year'] = df_retail_sales['Date'].dt.to_period('M')
df_property.drop(columns=['sale_date'], inplace=True)
df_retail_sales.drop(columns=['Date'], inplace=True)
merged_df = pd.merge(df_property, df_retail_sales, on='Month_Year', how='left')

merged_df = merged_df[merged_df['noi'] <= 5000000]

merged_df['Food and beverage stores'] = pd.to_numeric(merged_df['Food and beverage stores'], errors='coerce')
merged_df['Grocery stores'] = pd.to_numeric(merged_df['Grocery stores'], errors='coerce')
merged_df['Supermarkets and other grocery (except convenience) stores'] = pd.to_numeric(merged_df['Supermarkets and other grocery (except convenience) stores'], errors='coerce')

X = merged_df.drop(['noi', 'price', 'total_sf', 'cap_rate', 'Month_Year', 'Supermarkets and other grocery (except convenience) stores', 'property_name', 'city', 'state', '10_year_treasury', 'gdp', 'federal_funds_rate', 'interest_rate', 'Food and beverage stores', 'Grocery stores'], axis=1)
#  'multi_single_tenant',
y = merged_df['noi']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
summary = model.summary()
predictions = model.predict(X)

st.title("NOI Estimator")

st.sidebar.header("Input Property Details")

year_sold = st.sidebar.number_input("Year Sold", min_value=2000, max_value=2024, value=2010)
price_per_sf = st.sidebar.number_input("Price Per SF", value=147.66)
price = st.sidebar.number_input("Price", value=47163836.0)
# total_sf = st.sidebar.number_input("Total SF", value=319401)
multi_single_tenant = st.sidebar.selectbox("Multi/Single Tenant", options=[0, 1])
inflation = st.sidebar.number_input("Inflation Rate", value=4.69)

input_data = {
    'const': 1,
    'year_sold': year_sold,
    'price_per_sf': price_per_sf,
    # 'price': price,
    # 'total_sf': total_sf,
    'multi_single_tenant': multi_single_tenant,
    'inflation': inflation,
}

input_df = pd.DataFrame([input_data])
noi_prediction = model.predict(input_df)[0]

st.write(f"Predicted NOI: ${noi_prediction:.2f}")
property_price = price
adjusted_cap_rate = (noi_prediction / property_price) 
st.write(f"Adjusted Cap Rate: {adjusted_cap_rate:.2%}")

st.subheader("Distribution Graphs")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(merged_df['noi'], bins=30, alpha=0.7, label='NOI Distribution')
axes[0].axvline(noi_prediction, color='r', linestyle='dashed', linewidth=2, label='Predicted NOI')
axes[0].set_title('NOI Distribution')
axes[0].legend()

adjusted_cap_rates = merged_df['noi'] / merged_df['price']
axes[1].hist(adjusted_cap_rates, bins=30, alpha=0.7, label='Cap Rate Distribution')
axes[1].axvline(adjusted_cap_rate, color='r', linestyle='dashed', linewidth=2, label='Adjusted Cap Rate')
axes[1].set_title('Cap Rate Distribution')
axes[1].legend()

st.pyplot(fig)
