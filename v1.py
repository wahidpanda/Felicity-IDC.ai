import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Web App Title
st.markdown('''
# **The EDA App**

This is the **EDA App** created in Streamlit using the **pandas-profiling** library.

**Credit:** App built in `Python` + `Streamlit` by [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))

---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
    
    # Custom Data Visualizations
    st.header('**Custom Data Visualizations**')
    
    # Choose plot type
    plot_type = st.selectbox("Choose a plot type", ["Histogram", "Scatter Plot", "Box Plot", "Heatmap"])

    # For Histogram
    if plot_type == "Histogram":
        column = st.selectbox("Select column for histogram", df.columns)
        bins = st.slider("Number of bins", 5, 50, 20)
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=bins)
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # For Scatter Plot
    elif plot_type == "Scatter Plot":
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", df.columns)
        fig = px.scatter(df, x=x_axis, y=y_axis)
        fig.update_layout(title=f'Scatter Plot of {x_axis} vs {y_axis}', xaxis_title=x_axis, yaxis_title=y_axis)
        st.plotly_chart(fig)

    # For Box Plot
    elif plot_type == "Box Plot":
        column = st.selectbox("Select column for box plot", df.columns)
        fig, ax = plt.subplots()
        sns.boxplot(y=df[column], ax=ax)
        ax.set_title(f'Box Plot of {column}')
        st.pyplot(fig)

    # For Heatmap
    elif plot_type == "Heatmap":
        st.write("Heatmap of the correlation matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info('Awaiting for CSV file to be uploaded.')
