import argparse
import streamlit as st
import random
import matplotlib.pyplot as plt
import plotly.express as px

import os
from glob import glob
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from pprint import pprint, pformat
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

def main(args):
    df = pd.read_csv(args.data_file)
    class_list = sorted(list(df['class'].unique()))

    target_class = st.sidebar.selectbox('select class', class_list)
    target_df = df[df['class']==target_class]

    num_max_images = st.sidebar.number_input('images per page', min_value=1, max_value=len(target_df), value=10)

    st.image(random.sample(list(target_df['image_path'].values), num_max_images), width=128, use_column_width='never')

    fig = px.bar(df["class"].value_counts())
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    args = parser.parse_args()
    main(args)