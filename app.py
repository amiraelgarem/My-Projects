#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Streamlit App
# Title and About Section
st.title("Laptop Price Prediction Application")
st.sidebar.title("About the Project")
st.sidebar.write("""**Author**: Amira El-Garem""")
st.sidebar.write("""**Role**: BI-Developer""")
st.sidebar.write("""**Objective**: Predict laptop prices based on specifications like RAM, CPU, GPU, and more using machine learning models.
""")

# Tabs for EDA, Insights, and Prediction
tab1, tab2, tab3 = st.tabs(["EDA", "Insights", "Predict Price"])

# Load Data


data = pd.read_csv('laptop_price.csv')


import re 
def convert_memory(memory_str):
    total_memory = 0
    # Find all memory sizes in the string
    memory_matches = re.findall(r'(\d+)\s*(GB|TB)', memory_str)
    
    for value, unit in memory_matches:
        value = int(value)
        if unit == 'TB':
            total_memory += value * 1024  # Convert TB to GB
        else:
            total_memory += value  # Already in GB
    
    return total_memory


data['Memory'] = data['Memory'].apply(convert_memory)



def extract_resolution(resolution_str):
    match = re.search(r'(\d+)\s*x\s*(\d+)', resolution_str)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    return None, None

# Apply the function and create new columns
data[['Width', 'Height']] = data['ScreenResolution'].apply(lambda x: pd.Series(extract_resolution(x))) #check

# Calculate total pixels (optional)
data['Total Pixels'] = data['Width'] * data['Height']

columns_to_drop = [ 'Product','CPU_Company','CPU_Type','GPU_Company','GPU_Type','ScreenResolution', 'Width', 'Height' ]

data = data.drop(columns=columns_to_drop)


# 2. Descriptive Statistics
#print("\nDescriptive Statistics:\n", data.describe(include='all'))  # Replace with your file path


# Tab 1: EDA
with tab1:
    st.header("Exploratory Data Analysis")

    # Display dataset shape
    st.write("Shape of the dataset:", data.shape)

    st.write("Here is a snapshot of the dataset:")
    st.dataframe(data.head())

     # Check for null values
    st.subheader("Null Values Check")
    null_values = data.isnull().sum()
    if null_values.sum() == 0:
        st.write("No null values found in the dataset.")
    else:
        st.write("Number of null values in each column:")
        st.write(null_values[null_values > 0])
    


    # Display data types
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
    "Features": data.dtypes.index,
    "Data Type": data.dtypes.values
    })

    st.write(dtypes_df)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(numeric_only = True), annot=True, cmap="coolwarm", ax=ax).set(title='Heatmap of Correlation between Features')
    st.pyplot(fig)
    st.write('The heatmap reveals strong correlations between certain features. For example, price is positively correlated with RAM, CPU frequency, and weight. Notably, screen size and resolution do not show strong correlations with price.')

    # Visualization 1: Price Distribution
    st.subheader("Price Distribution")
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Price (Euro)'], bins=30, kde=True)
    plt.title('Price Distribution')
    st.pyplot(plt)
    st.write("The price distribution shows a right-skewed histogram, indicating that most laptops are priced lower, with fewer high-end models driving the average price up.")

    st.subheader("Price vs RAM")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='RAM (GB)', y='Price (Euro)', data=data, hue='Company', alpha=0.7)
    plt.title(f'Price vs RAM (Color-coded by Company)')
    plt.xlabel('RAM (GB)')
    plt.ylabel('Price (Euro)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    st.pyplot(plt)
    st.write("The scatter plot suggests a positive correlation between RAM and price, with higher RAM configurations typically associated with higher prices. Different companies are represented by color, showing brand-specific pricing strategies.")

    
with tab2:
    st.header("Randomforest Model Insights")
    loaded_model = pickle.load(open("random_forst_model_laptopprices.pkl", "rb"))
    data = pd.read_csv('transformed_laptop_prices_dataset.csv')
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    X = data.drop(['Price (Euro)'], axis=1)
    y = data['Price (Euro)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    loaded_model.fit(X_train, y_train)
    y_pred = loaded_model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    model_score = loaded_model.score(X_test, y_test)
    
    st.write("### Model Performance")
    st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.2f}")
    st.write(f"**Model Accuracy**: {(model_score *100):.2f}%")
 
     # Feature Importance
    st.subheader("Feature Importance")
    feature_importances = loaded_model.feature_importances_
    features = list(X.columns)
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    ax.set_ylabel('')
    st.pyplot(fig)

    # Predicted vs Actual Plot
    st.subheader("Predicted vs Actual Prices")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title("Predicted vs Actual Prices")
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    st.pyplot(fig)

# Tab 3: Predict Price
with tab3:
    st.header("Predict Laptop Price")
    st.write("Input the features of the laptop:")

# Load your model (replace 'your_model.joblib' with your actual file path)
    loaded_model = pickle.load(open("random_forst_model_laptopprices.pkl", "rb"))

    brand_mapping = {
        0: "Acer", 1: "Apple", 2: "Asus", 3: "Chuwi", 4: "Dell", 5: "Fujitsu",
        6: "Google", 7: "HP", 8: "Huawei", 9: "LG", 10: "Lenovo", 11: "MSI",
        12: "Mediacom", 13: "Microsoft", 14: "Razer", 15: "Samsung", 16: "Toshiba",
        17: "Vero", 18: "Xiaomi"
    }

    typename_mapping = {
        0: "2 in 1 Convertible", 1: "Gaming", 2: "Netbook", 3: "Notebook",
        4: "Ultrabook", 5: "Workstation"
    }

    opsys_mapping = {
        0: "Android", 1: "Chrome OS", 2: "Linux", 3: "Mac OS X", 4: "No OS",
        5: "Windows 10", 6: "Windows 10 S", 7: "Windows 7", 8: "macOS"
    }

    gpu_company_mapping = {
        0: "AMD", 1: "Intel", 2: "NVIDIA", 3: "Other"
    }

    gpu_type_mapping = {
        0: "GTX", 1: "RTX", 2: "Vega", 3: "MX", 4: "Integrated", 
        5: "Quadro", 6: "Radeon", 7: "Iris", 8: "UHD", 9: "HD Graphics", 
        10: "GeForce", 11: "NVS", 12: "FirePro", 13: "Tesla", 14: "Other"
    }

    selected_brand_name = st.selectbox(
        "Select Laptop Brand",
        options=list(brand_mapping.values())
    )

    # Convert the selected brand name back to its encoded value
    selected_brand_encoded = {v: k for k, v in brand_mapping.items()}[selected_brand_name]

    selected_typename_name = st.selectbox(
    "Select Laptop Type",
    options=list(typename_mapping.values())
    )
    selected_typename_encoded = {v: k for k, v in typename_mapping.items()}[selected_typename_name]

    selected_opsys_name = st.selectbox(
        "Select Operating System",
        options=list(opsys_mapping.values())
    )
    selected_opsys_encoded = {v: k for k, v in opsys_mapping.items()}[selected_opsys_name]

    selected_gpu_company_name = st.selectbox(
        "Select GPU Manufacturer",
        options=list(gpu_company_mapping.values())
    )
    selected_gpu_company_encoded = {v: k for k, v in gpu_company_mapping.items()}[selected_gpu_company_name]

    selected_gpu_type_name = st.selectbox(
        "Select GPU Type",
        options=list(gpu_type_mapping.values())
    )
    selected_gpu_type_encoded = {v: k for k, v in gpu_type_mapping.items()}[selected_gpu_type_name]

    inches = float(st.number_input("Inches", min_value=0.9, value=3.6))
    cpufrequency = float(st.number_input("CPU_Frequency (GHz)", min_value=10.1, value=18.4))
    ram = float(st.number_input("RAM (GB)", min_value=2, max_value=64, value=64))
    memory = float(st.number_input("Memory (GB)", min_value=0, value=2560))
    user_submit = st.button("Predict")

    user_data = {
    "Inches": [inches],
    'CPU_Frequency (GHz)': [cpufrequency],
    'RAM (GB)': [ram],
    'Memory': [memory],
    'Weight (kg)' : [2.04],
    'Total Pixels' : [2.073600e+06],
    'company' : [selected_brand_encoded],
    'typeName': [selected_typename_encoded],
    'operating_system': [selected_opsys_encoded],
    'gpu_manu' : [selected_gpu_company_encoded],
    'gpu_type' : [selected_gpu_type_encoded]
    
}
    if user_submit:
        user_data = pd.DataFrame.from_dict(user_data)
    # Prepare user data for prediction
        prediction = loaded_model.predict(user_data)
    
        st.subheader("Prediction Result:")
    
    
        
        st.write(f"Predicted Laptop Price: €{prediction[0]:,.2f}")
        




    st.markdown(
        """
        **Thank you for visiting!**  
        🌟 *Merry Laptop Shopping* 🌟  
        👉 [Connect on LinkedIn](https://www.linkedin.com/in/amira-el-garem-%F0%9F%87%B5%F0%9F%87%B8-aa7154138/)
        """
    )

