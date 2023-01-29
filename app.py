"""
Recommendations Engine 
Dibyendu Patra(mynamedibyendupatra@gmail.com)
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Recommendations",page_icon="Screenshot_10.png"
)
col_logo,col_header=st.columns([4,11])

with col_header:
    st.header("Dress Recommendations")


with st.sidebar:
    st.title("Reccomendations")
    recmd_selection=True


# Utility Functions which we will use through the rest of the workshop.

ss=st.session_state

if "images" not in ss:
    ss["images"]=[]
    ss["training"]=True
    ss["tfidf"]=None
    ss["tfidf_title"]=None
    ss["data"]=None
    ss["location"]=[]
    ss["reco_images"]={}
    ss["title"]=[]

#Display an image
def display_img(url,ax,fig):
    # we get the url of the apparel and download it
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # we will display it in notebook 
    ss["images"].append(img)

def cnn_image_open(url):
    # we get the url of the apparel and download it
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # we will display it in notebook 
    return img
  
#plotting code to understand the algorithm's decision.
def plot_heatmap(keys, values, labels, url, text):
        # keys: list of words of recommended title
        # values: len(values) ==  len(keys), values(i) represents the occurence of the word keys(i)
        # labels: len(labels) == len(keys), the values of labels depends on the model we are using
                # if model == 'bag of words': labels(i) = values(i)
                # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
                # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
        # url : apparel's url

        # we will devide the whole figure into two parts
        gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1]) 
        fig = plt.figure(figsize=(25,3))
        
        # 1st, ploting heat map that represents the count of commonly ocurred words in title2
        ax = plt.subplot(gs[0])
        # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
        ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
        ax.set_xticklabels(keys) # set that axis labels as the words of title
        ax.set_title(text) # apparel title
        
        # 2nd, plotting image of the the apparel
        ax = plt.subplot(gs[1])
        # we don't want any grid lines for image and no labels on x-axis and y-axis
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # we call dispaly_img based with paramete url
        display_img(url, ax, fig)
        
        # displays combine figure ( heat map and image together)
        
    
def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):

    intersection = set(vec1.keys()) & set(vec2.keys()) 

    for i in vec2:
        if i not in intersection:
            vec2[i]=0

    keys = list(vec2.keys())
    values = [vec2[x] for x in vec2.keys()]
    
    
    if model == 'tfidf':
        labels = []
        for x in vec2.keys():
            # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
            if x in  tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
    
    plot_heatmap(keys, values, labels, url, text)


# this function gets a list of wrods along with the frequency of each 
# word given "text"
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    # words stores list of all words in given string, you can try 'words = text.split()' this will also gives same result
    return Counter(words) # Counter counts the occurence of each word in list, it returns dict type object {word1:count}



def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b
    
    # vector1 = dict{word11:#count, word12:#count, etc.}
    vector1 = text_to_vector(text1)

    # vector1 = dict{word21:#count, word22:#count, etc.}
    vector2 = text_to_vector(text2)

    plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)


def tfidf_model(doc_id, num_results):
    # doc_id: apparel's id in given corpus
    
    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    pairwise_dist = pairwise_distances(tfidf_title_features,lavda[doc_id])

    # np.argsort will return indices of 9 smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the 9 smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])
    ss["images"]=[]
    ss["location"]=[]
    ss["title"]=[]
    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
        get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'tfidf')
        ASIN=data['asin'].loc[df_indices[i]]
        count=0
        title=data['title'].loc[df_indices[i]]
        ss["title"].append(title)

        for j in data["asin"]:
            if j==ASIN:
                ss["location"].append(count)
                break
            count=count+1
        
        

def get_similar_products_cnn(doc_id, num_results):
    bottleneck_features_train = np.load('16k_data_cnn_features.npy')
    asins = np.load('16k_data_cnn_feature_asins.npy')
    data = pd.read_pickle('16k_apperal_data_preprocessed')
    df_asins = list(data['asin'])
    asins = list(asins)

    doc_id = asins.index(df_asins[doc_id])
    pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    if "demo" in ss:
        ss["demo"]=[]
    for i in range(len(indices)):
        rows = data[['medium_image_url','title']].loc[data['asin']==asins[indices[i]]]
        for indx, row in rows.iterrows():
            df=ss["reco_images"]
            if "demo" not in ss:
                ss["demo"]=[]
            url=row['medium_image_url']
            img=cnn_image_open(url)
            ss["demo"].append(img)
            

def cnn(doc_id,num_results):
    get_similar_products_cnn(doc_id,num_results)

if recmd_selection:
    if ss["training"]:
        data = pd.read_pickle('16k_apperal_data_preprocessed')
        tfidf_title_vectorizer = TfidfVectorizer(min_df = 0)
        tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])
        ss["tfidf"]=tfidf_title_vectorizer
        ss["tfidf_title"]=tfidf_title_features
        ss["data"]=data
        ss["training"]=False
    z=st.text_input("Product Name ",
        key="placeholder",)
    number_of_Product=st.slider('Number of Product', 0, 40, 15)
    search=st.button("Search")

    if search:
        lists=[]
        lists.append(z)
        ds=pd.DataFrame({"check":lists})
        tfidf_title_features=ss["tfidf_title"]
        data=ss["data"]
        tfidf_title_vectorizer=ss["tfidf"]
        lavda=tfidf_title_vectorizer.transform(ds["check"])
        tfidf_model(0, number_of_Product)
        view_images=ss["images"]
        count=0
        se=ss["location"]

        for i in view_images:
            st.image(i)
            st.write("Title:",ss["title"][count])
            get_similar_products_cnn(se[count],6)
            id_=se[count]
            with st.expander("View Similar Products of "+ss["title"][count],expanded=False):
                recs=ss["reco_images"]
                n=6
                groups=[]
                view_images=ss["demo"]
                for i in range(0,len(view_images),n):
                    groups.append(view_images)
                for group in groups:
                    cols=st.columns(n)
                    for i,image in enumerate(group):
                        cols[i].image(image)
                
            count=count+1




                



