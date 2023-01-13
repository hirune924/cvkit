import argparse
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

@st.cache()
def load_csv(file_name):
    return pd.read_csv(file_name)

@st.cache()
def calc_metadata(df):
    height = np.zeros(len(df))
    width = np.zeros(len(df))
    aspect = np.zeros(len(df))
    mean_val = np.zeros(len(df))
    var_val = np.zeros(len(df))
    max_val = np.zeros(len(df))
    min_val = np.zeros(len(df))

    for idx, path in enumerate(df['image_path']):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height[idx], width[idx] = img.shape[:2]
        aspect[idx] = height[idx] / width[idx]
        mean_val[idx], var_val[idx], max_val[idx], min_val[idx] = img.mean(), img.var(), img.max(), img.min()
    df['height'] = height
    df['width'] = width
    df['aspect'] = aspect
    df['mean'] = mean_val
    df['var'] = var_val
    df['max'] = max_val
    df['min'] = min_val
    return df

def main(args):
    mode = st.selectbox('Mode select', ['Show dataframe', 'Show images', 'meta profile'])
    df = load_csv(args.data_file)
    if mode == 'Show dataframe':
        cols = st.columns(2)
        num_labeled = df['labeled'].sum()
        num_unlabeled = len(df) - num_labeled
        num_labeled_unique_class = len(df[df['labeled']==1]['class'].unique())
        num_labeled_unique_class_fold = \
            [len(df[(df['labeled']==1)&(df['fold']==i)]['class'].unique()) for i in df['fold'].unique()]
        with cols[0]:
            st.metric('Number of labeled', f'{num_labeled} ')
            st.metric('Number unique classes in labeled', f'{num_labeled_unique_class} ')
        with cols[1]:
            st.metric('Number of Unlabeled', f'{num_unlabeled} ')
        st.write(df)
        fig = px.bar(df["class"].value_counts(), title='Number of images per classes')
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")

        fig = px.bar(x=range(len(num_labeled_unique_class_fold)), y=num_labeled_unique_class_fold, title='Number of classes per fold')
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")

    elif mode == 'Show images':
        class_list = sorted(list(df['class'].dropna().unique()))
        cols = st.columns(2)
        with cols[0]:
            target_class = st.selectbox('select class', class_list)

        target_df = df[df['class']==target_class]
        target_df = target_df.sample(n=min(100, len(target_df)), frac=None, replace=False)
        with cols[1]:
            if st.button("shuffle", key=0):
                target_df = target_df.sample(n=min(100, len(target_df)), frac=None, replace=False)

        st.image(list(target_df['image_path'].values), width=128, use_column_width='never')

    elif mode == 'meta profile':
        meta_df = calc_metadata(df.copy())
        pr = meta_df.profile_report()

        st_profile_report(pr)


if __name__ == "__main__":
    st.set_page_config(layout='wide')
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    args = parser.parse_args()
    main(args)