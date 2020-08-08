import urllib
import pickle
import pathlib
import enum

import dateparser
import imageio
import pandas as pd
import streamlit as st
from loguru import logger
import nltk
from nltk.corpus import wordnet as wn
from urllib.parse import quote
import numpy as np

import plotter

class ModelTypes(enum.Enum):
    MODEL = ('Model trained on ImageNet', 'tab:blue', 150)
    MODELPLUS = ('Model trained on more data', 'tab:red', 150)
    HUMAN = ('Human labelers', 'tab:green', 150)
    HUMAN_ENSEMBLE = ('Human Ensemble', 'tab:purple', 150)

def get_user_type(df_row):
    user_name = df_row.username.lower()
    MODELSPLUS = ["fixresnext101_32x48d_v2", "instagram-48d"]
    if user_name == "human_majority":
        return ModelTypes.HUMAN_ENSEMBLE
    elif "human" in user_name:
        return ModelTypes.HUMAN
    elif user_name in MODELSPLUS:
        return ModelTypes.MODELPLUS
    else:
        return ModelTypes.MODEL

def add_dataset_size(df_row):

    if df_row.dataset == "v2":
        return v2_size
    elif df_row.dataset == "val":
        return val_size
    else:
        assert False

def show_in_plot(df_row):
    if "fv" not in df_row.username and df_row.val > 70 and df_row.username != "model_majority" and df_row.username != "instagram_model_48d":
        return True

def use_for_line_fit(df_row):
    user_name = df_row.username.lower()
    return "human" not in user_name

def load_data():
    human_accuracy_data = pathlib.Path("human_accuracy.pickle")

    if not human_accuracy_data.exists():
        download_state = st.text("Downloading human_accuracy.pickle...")
        data_bytes = urllib.request.urlopen("https://pictureweb.s3-us-west-2.amazonaws.com/human_accuracy.pickle").read()
        download_state = st.text("Downloading human_accuracy.pickle...done")
        with open("human_accuracy.pickle", "wb") as f:
            f.write(data_bytes)

    with open("human_accuracy.pickle", "rb") as f:
        ha_data = pickle.load(f)
    return load_ha(ha_data)

def gen_acc_df(h_a):
    all_data = []
    for user in h_a.annot_data.keys():
        try:
            all_data.append(compute_acc_dict(h_a, user, extra_filters=[], top1=False))
        except KeyError as e:
            pass
        acc_df = pd.DataFrame(all_data, columns=["username", "val", "v2"])
        acc_df = acc_df.rename(columns={'val': 'ImageNet Multi-Label Accuracy', 'v2': 'ImageNetV2 Multi-Label Accuracy', 'username': "Classifier"})

    return acc_df


def is_wnid(wnid):
    try:
        wn.synset_from_pos_and_offset(wnid[0],int(wnid[1:]))
        return True
    except:
        return False

def get_wnids(name):
    return ['n0' + str(x.offset()) for x in wn.synsets(name)]


def imageid_to_key(image_id):
    if image_id.startswith('ILSVRC2012_val'):
        return f'imagenet_validation_flat/{image_id}', "ImageNet-Val"
    else:
        return f'imagenet2candidates_original/{image_id}.jpg', "ImageNetV2-Matched-Frequency"

def wnid_to_name(wnid):
    synset = wn.synset_from_pos_and_offset(wnid[0],int(wnid[1:]))
    return synset.lemmas()[0].name()


def s3_key_to_url(key, bucket="imagenet2datav2"):
    return f"https://s3-us-west-2.amazonaws.com/{bucket}/{quote(key)}"

def make_fig(h_a, df, fig_title, filter_fn=None):
    if filter_fn is None:
        filter_fn = h_a.NO_FILTER

    def filter_df_row(df_row):
        username = df_row['username']
        acc_val, num_img_filter_val = h_a.compute_accuracy(username, filters=[filter_fn, h_a.VAL_SET, h_a.NOT_PROBLEMATIC])
        acc_v2, num_img_filter_v2 = h_a.compute_accuracy(username, filters=[filter_fn, h_a.VTWO_SET, h_a.NOT_PROBLEMATIC])

        df_row['val'] = 100 * acc_val
        df_row['v2'] = 100 * acc_v2

        df_row['val_dataset_size'] = num_img_filter_val
        df_row['v2_dataset_size'] = num_img_filter_v2

        return df_row

    df_filter = df.copy()
    df_filter = df_filter.apply(filter_df_row, axis = 1)


    x_axis, y_axis = 'val', 'v2'

    df1 = plotter.add_plotting_data(df_filter, [x_axis, y_axis])

    # auto set xlim and ylim based on visible points
    df_visible = df1[df1.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 2, df_visible[x_axis].max() + 2]
    ylim = [df_visible[y_axis].min() - 2, df_visible[[x_axis, y_axis]].values.max() +2]
    fig, _ = plotter.model_scatter_plot(df1, x_axis, y_axis, xlim, ylim, ModelTypes,
                                                num_bootstrap_samples=10, transform='linear', tick_multiplier=5,
                                                title=fig_title, x_label='ImageNet', y_label='ImageNetV2',
                                                figsize=(12, 8), include_legend=True, return_separate_legend=False)

    table_users = ['human_a', 'human_b', 'human_c', 'human_d', 'human_e', 'FixResNeXt101_32x48d_v2',
                   'instagram-48d', 'efficientnet-b7', 'resnet50']

    for user in table_users:
        print(user, float(df1.loc[df1['username'] == user]['val']), float(df1.loc[df1['username'] == user]['v2']))

