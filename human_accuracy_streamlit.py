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


from human_accuracy import *
import plotter

def load_ha(ha_data):
    return HumanAccuracyPortable(annot_data=ha_data["annot_data"], 
                                 labeled_images=ha_data["labeled_images"], 
                                 image_labels=ha_data["image_labels"],
                                 split_wnids=ha_data["split_wnids"],
                                 image_map=ha_data["image_map"],
                                 all_candidates=ha_data["all_candidates"],
                                 wnid_map=ha_data["wnid_map"])

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

if __name__ == "__main__":
    st.title('Contextualizing Machine Accuracy Playground')

    nltk_download_state = st.text('Loading NLTK...')
    nltk.download('wordnet')
    nltk_download_state.text('Loading NLTk...done!')
    data_load_state = st.text('Loading data...')
    h_a = load_data()
    data_load_state.text('Loading data...done!')
    acc_df = gen_acc_df(h_a)
    st.markdown('## Multi-Label Accuracies')
    st.write(acc_df)
    st.markdown('## Explore Annotations')
    st.markdown('### Enter WordNet-ID or Class Name to explore annotations')
    class_name_or_wnid = st.text_input("", "paper towel")
    class_name_or_wnid = class_name_or_wnid.replace(" ", "_")
    logger.debug(f"Search query: {class_name_or_wnid}")
    if is_wnid(class_name_or_wnid):
        wnids = [class_name_or_wnid]
    else:
        wnids = get_wnids(class_name_or_wnid)

    wnids = [x for x in wnids if x in h_a.rev_wnid_map]
    if len(wnids) == 0:
        st.write("Invalid Query")
    else:
        imgs = []
        for wnid in wnids:
            for img in h_a.rev_wnid_map[wnid]:
                img_key, img_src = imageid_to_key(img)
                img_url = s3_key_to_url(img_key)
                multi_labels = [wnid_to_name(x) for x in h_a.image_labels[img]["correct"]]
                top_1_label = wnid_to_name(wnid)
                st.markdown("**Multi-Labels**: " + ",".join(multi_labels))
                st.markdown("**ImageNet Top-1 Label**: " + top_1_label)
                st.image(img_url, caption=f"Image Source: {img_src}", use_column_width=True)









