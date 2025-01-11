import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#RUNCOMMAND
#streamlit run C:\Users\thoma\PycharmProjects\Project_OBP\dashboard.py

#installs:
#pip install pandas
#pip install streamlit
#pip install openpyxl

st.set_page_config(page_title = 'Collaboration Dashboard', page_icon = ":bar_chart", layout = 'wide')
st.title(" :bar_chart: Collaboration Dashboard")

st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

# File uploader
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))

if fl is not None:
    # Use the file-like object directly
    if fl.name.endswith('.csv'):
        df = pd.read_csv(fl, encoding="ISO-8859-1")
    elif fl.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(fl, engine='openpyxl')
    else:
        st.error("Unsupported file type!")
        st.stop()

    st.write(f"Uploaded file: {fl.name}")
    st.dataframe(df)  # Display the dataframe






