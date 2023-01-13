import argparse
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2

@st.cache()
def load_csv(file_name):
    return pd.read_csv(file_name)

def main(args):
    mode = st.selectbox('Mode select', ['Show dataframe', 'Show high score images', 'Show images'])
    df = load_csv(args.data_file)
    if mode == 'Show dataframe':
        classes = sorted(list(df['class'].unique()))
        st.write(df)
        fig = px.bar(df["class"].value_counts()[classes],title='Number of grandtruth classes')
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")

        loss_cols = df.columns[df.columns.str.contains('loss_[0-9]')]
        group_df = df.groupby('class').mean()[loss_cols]
        plot_df = pd.concat([pd.DataFrame.from_dict({'epoch': list(range(len(loss_cols))),
            'loss': group_df.loc[c,:].values, 'breed': [c]*len(loss_cols)})
            for c in classes])
        fig = px.line(plot_df, x='epoch', y='loss',color='breed', title='loss transition')
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")

        group_df = df.groupby('class').mean()[['loss_avg', 'loss_var']]
        fig = px.bar(group_df,title='Normalized loss per class')
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
        
    elif mode == 'Show high score images':
        number = st.number_input('Display number', min_value=1, value=20)
        target_df = df.sort_values('loss_avg', ascending=False)[:number]
        with st.expander("Whole data", expanded=True):
            st.markdown("""---""")
            for idx, item in target_df.iterrows():

                cols = st.columns(2)
                with cols[0]:
                    st.image(item['image_path'], width=256, use_column_width='never')
                with cols[1]:
                    st.metric('Grand truth', f"{item['class']}")
                    st.metric('O2U Score', f"{item['loss_avg']:.4f}")
                st.markdown("""---""")

        with st.expander("Specified class", expanded=True):
            class_list = sorted(list(df['class'].unique()))
            target_class = st.selectbox('select class', class_list)
            target_df = df[df['class']==target_class].sort_values('loss_avg', ascending=False)[:number]
            st.markdown("""---""")
            
            for idx, item in target_df.iterrows():

                cols = st.columns(2)
                with cols[0]:
                    st.image(item['image_path'], width=256, use_column_width='never')
                with cols[1]:
                    st.metric('O2U Score', f"{item['loss_avg']:.4f}")
                st.markdown("""---""")

    elif mode == 'Show images':
        class_list = sorted(list(df['class'].unique()))
        cols = st.columns(2)
        with cols[0]:
            target_class = st.selectbox('select class', class_list)

        target_df = df[df['class']==target_class]
        target_df = target_df.sample(n=min(100, len(target_df)), frac=None, replace=False)
        with cols[1]:
            if st.button("shuffle", key=0):
                target_df = target_df.sample(n=min(100, len(target_df)), frac=None, replace=False)

        st.image(list(target_df['image_path'].values), width=128, use_column_width='never')



if __name__ == "__main__":
    st.set_page_config(layout='wide')
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    args = parser.parse_args()
    main(args)