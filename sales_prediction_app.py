import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# Set Streamlit page config
st.set_page_config(page_title="📊 Sales Prediction App", layout="centered")
st.title("📊 Sales Prediction Using Advertising Budget")
st.markdown("This app predicts product sales based on **TV**, **Radio**, and **Newspaper** advertising budgets using a **Linear Regression Model**.")

# Embed CSV dataset directly inside the app
@st.cache_data
def load_data():
    data = """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
8.7,48.9,75.0,7.2
57.5,32.8,23.5,11.8
120.2,19.6,11.6,13.2
8.6,2.1,1.0,4.8
199.8,2.6,21.2,10.6
66.1,5.8,24.2,8.6
214.7,24.0,4.0,17.4
23.8,35.1,65.9,9.2
97.5,7.6,7.2,9.7
204.1,32.9,46.0,19.0
195.4,47.7,52.9,22.4
67.8,36.6,114.0,12.5
281.4,39.6,55.8,24.4
69.2,20.5,18.3,11.3
147.3,23.9,19.1,14.6"""
    return pd.read_csv(StringIO(data))

df = load_data()

# Show dataset
with st.expander("🔍 View Dataset"):
    st.dataframe(df)

# Visualization
with st.expander("📈 Show Exploratory Data Analysis"):
    st.subheader("Pairplot: Ad Spend vs Sales")
    fig = sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg', height=4)
    st.pyplot(fig)

# Model training
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")

# Input form
st.subheader("🧠 Predict Sales Based on Ad Spend")
tv = st.slider("TV Advertising Budget ($)", 0.0, 300.0, 150.0)
radio = st.slider("Radio Advertising Budget ($)", 0.0, 50.0, 25.0)
newspaper = st.slider("Newspaper Advertising Budget ($)", 0.0, 120.0, 20.0)

# Prediction
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
prediction = model.predict(input_data)[0]
st.success(f"📈 Predicted Sales: **{prediction:.2f} units**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Meesam Raza** | Data Science Internship Project")
