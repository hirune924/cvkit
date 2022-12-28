import argparse
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from glob import glob
from sklearn import metrics

def main(args):
    target_list = glob(args.eval_regex)
    target_csv = st.multiselect('Target Eval result',target_list,target_list)
    df = pd.concat([pd.read_csv(p) for p in target_csv])
    classes = [c.replace('confidence_', '') for c in df.columns.values if 'confidence' in c ]

    mode = st.selectbox('Mode select', ['Show dataframe', 'Show metrics', 'Confusion matrix', 'Show images', 'Show hard samples'])
    if mode == 'Show dataframe':
        st.write(df)
        fig = px.bar(df["class"].value_counts())
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
        
    elif mode == 'Show metrics':
        report = metrics.classification_report(df['class_id'], df['predict'], target_names=classes, output_dict=True)
        accuracy = report.pop('accuracy')
        st.metric('Accuracy', f'{accuracy*100:.2f} %')
        st.table(pd.DataFrame(report).transpose().style.background_gradient(axis=0))

    elif mode == 'Confusion matrix':
        cm = metrics.confusion_matrix(df['class_id'], df['predict'])
        cm = pd.DataFrame(cm, index=classes, columns=classes)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predict", y="Grand truth"), width=800, height=800)
        st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

    elif mode == 'Show hard samples':
        conf_cols = [c for c in df.columns.values if 'confidence' in c ]
        hard_df = df[df['class_id']!=df['predict']] 
        target_class = st.selectbox('select class', classes)
        hard_df = hard_df[hard_df['class']==target_class]
        hard_df = hard_df.sample(n=min(10, len(hard_df)), frac=None, replace=False)
        if st.button("shuffle", key=0):
            hard_df = hard_df.sample(n=min(10, len(hard_df)), frac=None, replace=False)

        st.markdown("""---""")
        for idx, item in hard_df.iterrows():

            cols = st.columns(2)
            with cols[0]:
                st.image(item['image_path'], width=256, use_column_width='never')
            with cols[1]:
                gt = item['class']
                gt_score = item['confidence_'+gt]
                pred = classes[item['predict']]
                pred_score = item['confidence_'+pred]
                confidence = item[conf_cols]
                st.text(f'GT: {gt} (conf: {gt_score:.2f})')
                st.text(f'Predicted: {pred} (conf: {pred_score:.2f})')
                fig = px.bar(x=classes,y=confidence, range_y=[0.0, 1.0])
                st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")
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
    parser.add_argument('eval_regex', type=str)
    args = parser.parse_args()
    main(args)