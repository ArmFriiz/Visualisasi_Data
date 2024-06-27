import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.caption("Muhammad Faris Akbar | 2100018169")

sns.set(style="whitegrid")

# Membaca file CSV ke dalam DataFrame
data = pd.read_csv('https://raw.githubusercontent.com/ArmFriiz/Visualisasi_Data/main/Groceries_dataset.csv', delimiter=',')

# Streamlit application
st.title("Analisis dan Prediksi Penjualan Grocery")

st.subheader("Preview Data")
st.dataframe(data, width=2500, height=250)
st.write("-----------------------------------------------------------------------------------------------------------------")

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Calculate frequency of each item
item_counts = data['itemDescription'].value_counts()

# Calculate the number of unique members and items
unique_members = data['Member_number'].nunique()
unique_items = data['itemDescription'].nunique()

# Pengelompokan data berdasarkan hari dan bulan untuk digunakan pada analisis trend
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

monthly_trend = data.groupby('Month').size()
weekly_trend = data.groupby('DayOfWeek').size()

# Penambahan atribut timestamp untuk digunakan pada forecasting penjualan bulanan
data['Forecast_monthly'] = data['Date'].dt.to_period('M')
monthly_sales = data.groupby('Forecast_monthly').size()
monthly_sales.index = monthly_sales.index.to_timestamp()

# Define categories for items
categories = {
    'buah': ['tropical fruit', 'pip fruit', 'citrus fruit', 'berries', 'stone fruit', 'grapes', 'exotic fruit'],
    'sayur': ['other vegetables', 'root vegetables', 'herbs', 'canned vegetables', 'fresh vegetables', 'frozen vegetables'],
    'bahan_susu': ['whole milk', 'yogurt', 'butter', 'cream', 'curd', 'hard cheese', 'soft cheese', 'processed cheese', 'cheese', 'butter milk'],
    'daging': ['sausage', 'meat', 'poultry', 'fish', 'hamburger meat', 'frozen meals', 'frozen chicken', 'frozen fish', 'chicken', 'beef', 'frankfurter', 'pork'],
    'minuman': ['soda', 'bottled water', 'fruit/vegetable juice', 'frozen juice', 'canned beer', 'liquor', 'red/blush wine', 'white wine'],
    'snack': ['rolls/buns', 'pastry', 'cookies', 'candy', 'chocolate', 'popcorn', 'snack products'],
    'bahan_dapur': ['flour', 'sugar', 'salt', 'baking powder', 'spices', 'honey', 'vinegar', 'sauces', 'oil', 'long life bakery product'],
    'makanan_siap': ['instant coffee', 'tea', 'ready soups', 'canned fish', 'canned fruit', 'canned vegetables', 'packaged rice', 'packaged pasta', 'packaged fruit/vegetables'],
    'gandum': ['brown bread', 'waffles'],
    'hewan': ['cat food', 'dog food'],
    'aksesoris': ['pot plants', 'specialty bar', 'cling film/bags', 'kitchen towels', 'napkins'],
    'lainnya': []
}

# Create a new column for category
def categorize_item(item):
    for category, items in categories.items():
        if item in items:
            return category
    return 'lainnya'

data['Category'] = data['itemDescription'].apply(categorize_item)

category_groups = {
    'buah': data[data['Category'] == 'buah'],
    'sayur': data[data['Category'] == 'sayur'],
    'bahan_susu': data[data['Category'] == 'bahan_susu'],
    'daging': data[data['Category'] == 'daging'],
    'minuman': data[data['Category'] == 'minuman'],
    'snack': data[data['Category'] == 'snack'],
    'bahan_dapur': data[data['Category'] == 'bahan_dapur'],
    'makanan_siap': data[data['Category'] == 'makanan_siap'],
    'gandum': data[data['Category'] == 'gandum'],
    'hewan': data[data['Category'] == 'hewan'],
    'aksesoris': data[data['Category'] == 'aksesoris'],
    'lainnya': data[data['Category'] == 'lainnya']
}

monthly_sales_category = {
    category: group.groupby('Forecast_monthly').size() for category, group in category_groups.items()
}

# Menentukan kategori terfavorit untuk setiap bulan
monthly_category_counts = data.groupby(['Month', 'Category']).size().reset_index(name='Count')
monthly_favorites_category = monthly_category_counts.loc[monthly_category_counts.groupby('Month')['Count'].idxmax()]
monthly_least_favorites_category = monthly_category_counts.loc[monthly_category_counts.groupby('Month')['Count'].idxmin()]

# Menentukan item terfavorit untuk setiap bulan
monthly_item_counts = data.groupby(['Month', 'itemDescription']).size().reset_index(name='Count')
monthly_favorites_item = monthly_item_counts.loc[monthly_item_counts.groupby('Month')['Count'].idxmax()]
monthly_least_favorites_item = monthly_item_counts.loc[monthly_item_counts.groupby('Month')['Count'].idxmin()]

def forecast_arima(data, steps):
    model = ARIMA(data, order=(5, 1, 0))  # Order model dapat disesuaikan
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

st.subheader("Trend Pembelian Bulanan")
fig, ax = plt.subplots(figsize=(14, 3))
monthly_trend.plot(kind='line', marker='o', linestyle='-', color='b', ax=ax)
ax.set_title('Trend Pembelian Bulanan')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Pembelian')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
st.pyplot(fig)

st.subheader("Trend Pembelian Harian")
fig, ax = plt.subplots(figsize=(14, 3))
weekly_trend.plot(kind='line', marker='o', linestyle='-', color='g', ax=ax)
ax.set_title('Trend Pembelian Harian')
ax.set_xlabel('Hari')
ax.set_ylabel('Jumlah Pembelian')
ax.set_xticks(range(7))
ax.set_xticklabels(['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'])
st.pyplot(fig)

st.write("Berdasarkan data trend pembelian bulanan tersebut, didapatkan bahwa Bulan Agustus merupakan bulan yang paling banyak terjadi pembelian sedangkan untuk Februari dan Desember menjadi yang paling sedikit")
st.write("Data tersebut bisa diperkuat dengan menggunakan data trend pembelian harian, dengan adanya data pembelian harian stakeholder dapat memiliki gambaran dalam pengambilan keputusan untuk pengelolaan stock dan strategi pemasaran")

st.write("-----------------------------------------------------------------------------------------------------------------")

left,right= st.columns(2)

with left:
    st.subheader("Kategori Paling Diminati Per Bulan")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.barplot(data=monthly_favorites_category, x='Month', y='Count', hue='Category', dodge=False, ax=ax)
    ax.set_title('Kategori Paling Diminati Per Bulan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Frekuensi Penjualan')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
    st.pyplot(fig)

    st.subheader("Item Paling Diminati Per Bulan")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.barplot(data=monthly_favorites_item, x='Month', y='Count', hue='itemDescription', dodge=False, ax=ax)
    ax.set_title('Item Paling Diminati Per Bulan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Frekuensi Penjualan')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
    st.pyplot(fig)

with right:
    st.subheader("Kategori Kurang Diminati Per Bulan")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.barplot(data=monthly_least_favorites_category, x='Month', y='Count', hue='Category', dodge=False, ax=ax)
    ax.set_title('Kategori Kurang Diminati Per Bulan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Frekuensi Penjualan')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
    st.pyplot(fig)

    st.subheader("Item Kurang Diminati Per Bulan")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.barplot(data=monthly_least_favorites_item, x='Month', y='Count', hue='itemDescription', dodge=False, ax=ax)
    ax.set_title('Item Kurang Diminati Per Bulan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Frekuensi Penjualan')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
    st.pyplot(fig)

st.write("Data item serta Kategori yang paling diminati atau data item dan kategori yang kurang diminati dapat menjadi data tambahan yang berguna untuk stakeholder dalam pengambilan keputusan, data tersebut dapat membantu stakeholder dalam memfilter stock barang yang perlu ditingkatkan ataupun dikurangi")

st.write("-----------------------------------------------------------------------------------------------------------------")

st.subheader("Frekuensi Penjualan per Item")
fig, ax = plt.subplots(figsize=(14, 3))
item_counts.head(10).plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('10 Item Paling Diminati')
ax.set_xlabel('Item')
ax.set_ylabel('Frekuensi')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20, 5))
item_counts.tail(10).plot(kind='bar', color='lightcoral', ax=ax)
ax.set_title('10 Item Kurang Diminati')
ax.set_xlabel('Item')
ax.set_ylabel('Frekuensi')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
st.pyplot(fig)

st.write("Data frekuensi penjualan item yang paling diminati atau yang kurang diminati dapat menjadi data pendukung apabila visualisasi sebelumnya dianggap kurang jelas. Data ini dapat juga dijadikan sebagai pertimbangan stakeholder apabila ingin memfilter data barang secara spesifik berdasarkan minat pembeli")

st.write("-----------------------------------------------------------------------------------------------------------------")

st.subheader("Prediksi Penjualan 12 Bulan Ke Depan")
selected_category = st.selectbox("Pilih kategori", list(monthly_sales_category.keys()))

left_2,right_2= st.columns(2)

with left_2:
    st.subheader("Berdasarkan Bulan")
    monthly_forecast = forecast_arima(monthly_sales, 12)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly_sales.index, monthly_sales.values, label='Actual')
    ax.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), monthly_forecast, label='Forecast', color='red')
    ax.set_title('Prediksi Penjualan 12 Bulan Ke Depan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Jumlah Penjualan')
    ax.legend()
    st.pyplot(fig)
with right_2:
    st.subheader("Berdasarkan Kategori")
    for category, sales_data in monthly_sales_category.items():
        if selected_category == category:
            sales_data.index = sales_data.index.to_timestamp()
            forecast = forecast_arima(sales_data, 12)
        
            forecast_index = pd.date_range(sales_data.index[-1], periods=12, freq='M')

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(sales_data.index, sales_data.values, label='Actual')
            ax.plot(forecast_index, forecast, label='Forecast', color='red')
            ax.set_title(f'Prediksi Penjualan 12 Bulan Ke Depan: {category.capitalize()}')
            ax.set_xlabel('Bulan')
            ax.set_ylabel('Jumlah Penjualan')
            ax.legend()
            st.pyplot(fig)

st.write("Data prediksi merupakan data yang paling mungkin dapat digunakan apabila stakeholder menginginkan gambaran penjualan dan profit yang bisa didapatkan. Dengan data prediksi ini stakeholder setidaknya dapat memiliki perkiraan biaya dan jumlah barang yang harus dikelola")
