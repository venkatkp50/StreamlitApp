import io
import os
import re
import json
import time
import math
import zipfile
import logging
import requests
import openai
import rouge
import nltk
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st
from pprint import pprint
from copy import deepcopy
from tqdm.notebook import tqdm
from streamlit_chat import message
import seaborn as sns
import matplotlib.pyplot as plt
import re, os, string, random, requests
from subprocess import Popen, PIPE, STDOUT
from haystack.nodes import EmbeddingRetriever
from haystack.utils import clean_wiki_text
from haystack.utils import convert_files_to_docs
from haystack.utils import fetch_archive_from_http,print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from summarizer import Summarizer,TransformerSummarizer
from bert_score import score
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk import sent_tokenize, word_tokenize
from nltk.translate import meteor
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.set_page_config(layout="wide")
# run block of code and catch warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

imagename2 = Image.open('images/Sidebar2.jpg')
st.sidebar.image(imagename2)

#st.sidebar.markdown('-------')
st.sidebar.title('Settings')

modelSelected = st.sidebar.selectbox('Choose Reader Model',options=('deepset/roberta-base-squad2','deepset/roberta-base-squad2-covid','deepset/covid_bert_base'))
#read_top_k = st.sidebar.selectbox('Read Top N Choices',options=(5,7,10))


imagename = Image.open('images/caronavirus banner.jpg')
st.image(imagename)
st.text_input("Your Query", key="input_text",value='')
user_message = st.session_state.input_text

doc_dir = "txtfile1"
data = pd.read_csv('json_csv\pdf_json.csv')
document_store = InMemoryDocumentStore(use_bm25=True)
docs = convert_files_to_docs(dir_path=doc_dir,clean_func=clean_wiki_text,split_paragraphs=True)
document_store.write_documents(docs)
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path=modelSelected, use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)


if user_message != '':
    results = pipe.run(query=user_message,params={"Retriever": {"top_k": 10},"Reader": {"top_k": 5}})
    ans = []
    doc = []
    score = []
    context = []
    id =[]
    for result in results['answers']:
        ans.append(result.answer)
        score.append(result.score)
        context.append(result.context)
        id.append(result.meta['name'])
 
    responsedf = pd.DataFrame({'Probable Anwsers':ans,'Score':score,'Context':context,'Source File Name':id})
    # st.table(responsedf[['Probable Anwsers','Score','Context']])
    score100 = [scr*100 for scr in score]
    #colorcode = ['rgb(116, 191, 0)', 'rgb(60, 194, 0)', 'rgb(2, 198, 0)', 'rgb(0, 210, 186)', 'rgb(0, 174, 213)']
    colorcode = ['rgb(102, 0, 51)', 'rgb(204, 0, 102)', 'rgb(255, 51, 153)', 'rgb(102, 255, 255)', 'rgb(204, 204, 255)']
    opacitycode = [0.8, 0.6, 0.5, 0.4,0.3]
    fig = go.Figure(data=[go.Scatter(x=ans, y=score100,marker=dict(color=colorcode,opacity=opacitycode,size=score100,))])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True,)

    tab1, tab2, tab3 = st.tabs(["Summarization", "Author Details", ""])

    with tab1:

        

        n1file = open(doc_dir+'/'+responsedf.loc[0,'Source File Name'],"r")
        paper_id = responsedf.loc[0,'Source File Name'].replace('.txt','')
        golddf = data[data['paper_id']== paper_id]
        golddf_text = golddf['abstract'].values[0]
        summdf = pd.DataFrame({''})

        with open(doc_dir+'/'+responsedf.loc[0,'Source File Name'], "r") as fd:
            fulltext = fd.readlines()
        
        header =[]
        para = []
        for line in fulltext:
            if len(line) > 1:
                if len(line) < 100:
                    header.append(line)
                else:
                    para.append(line)   
        
        col1, col2, col3 = st.columns(3)

        st.write('1....')
        st.write(len(golddf_text))
        tot_words_ref = len(word_tokenize(golddf_text))
        col1.subheader("Reference Standard")
        col1.write(golddf_text)

        col2.subheader('BERT')
        bert_model = Summarizer()        
        body = n1file.read()     

        berttext = []
        for parabody in para:            
            berttext.append(bert_model(body=parabody,max_length=100))
        bert_summary = ''.join( lines for lines in berttext)
        bert_summary = bert_model(body=bert_summary,max_length=math.ceil(tot_words_ref / 100) * 100)
        col2.write(bert_summary)        
        col3.subheader('GPT-2')
        GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
        gpt2text = []
        for parabody in para:            
            gpt2text.append(GPT2_model(body=parabody, max_length=math.ceil(tot_words_ref / 100) * 100))
        #gpt2text_summary = ''.join(GPT2_model(body=gpt2text, max_length=math.ceil(tot_words_ref / 100) * 100)) 
        gpt2text_full = ''.join(text for text in gpt2text)
        gpt2text_summary = GPT2_model(body=gpt2text_full, max_length=math.ceil(tot_words_ref / 100) * 100)
        col3.write(gpt2text_summary)        

        st.markdown('---')   

        col1 , col2, col3 = st.columns(3)

        tot_words_bert = len((bert_summary.split()))
        tot_words_gpt3 = len((gpt2text_summary.split()))
        col1.metric('Total Words Reference Text',tot_words_ref)
        col2.metric("Total Words BERT Summarization", tot_words_bert,(tot_words_bert - tot_words_ref))
        col3.metric("Total Words GPT-3 Summarization",tot_words_gpt3,(tot_words_gpt3 - tot_words_ref) )

        tot_words_ref = len(sent_tokenize(golddf_text))
        tot_words_bert = len(sent_tokenize(bert_summary))
        tot_words_gpt3 = len(sent_tokenize(gpt2text_summary))
        
        col1.metric('Total Sentences Reference Text',tot_words_ref)
        col2.metric("Sentences in BERT Summarization", tot_words_bert,(tot_words_bert - tot_words_ref))
        col3.metric(" Sentences in GPT-2 Summarization",tot_words_gpt3,(tot_words_gpt3 - tot_words_ref) )

        st.markdown('---')   
        st.subheader('Performance Analysis of Text-Summary')     
        rouge = rouge.Rouge()
        bertscores = rouge.get_scores(hyps=golddf_text, refs=bert_summary, avg=True)        
        gpt2scores = rouge.get_scores(hyps=golddf_text, refs=gpt2text_summary, avg=True)   

        col1, col2, col3 = st.columns(3)
        col1.write('**Interpretation of Rouge Score**')
        col1.write('**Recall**=40% means that 40% of the n-grams in the reference summary are also present in the generated summary.')
        col1.write('**Precision**=40% means that 40% of the n-grams in the generated summary are also present in the reference summary.')
        col1.write('**F1-score**=40% is more difficult to interpret, like any F1-score.')
        
        col2.write('BERT Score')
        bertscore = pd.DataFrame(bertscores)
        col2.table(bertscore)

        col3.write('GPT-3 Score')
        gpt2score = pd.DataFrame(gpt2scores)
        col3.table(gpt2score)

        col1, col2, col3 = st.columns(3)
        col2.write('BERT ROUGE CHART')
        col2.bar_chart(bertscore)
        col3.write('GPT-2 ROUGE CHART')
        col3.bar_chart(gpt2score)

       
        st.markdown('---')          

        target  = ['BERT','GPT-3']
        r1 = [bertscore.loc['f','rouge-1'],gpt2score.loc['f','rouge-1']]
        r2 = [bertscore.loc['f','rouge-2'],gpt2score.loc['f','rouge-2']]
        r3 = [bertscore.loc['f','rouge-l'],gpt2score.loc['f','rouge-l']]
        refs = [sent.split() for sent in sent_tokenize(golddf_text)]
        cands  = bert_summary.split()
        bert_beluscore = sentence_bleu(refs, cands)
        cands  = gpt2text_summary.split()
        gpt_beluscore = sentence_bleu(refs, cands)
        belu = [ bert_beluscore,gpt_beluscore]

        bert_metor = meteor([word_tokenize(golddf_text)],word_tokenize(bert_summary))
        gpt_metor = meteor([word_tokenize(golddf_text)],word_tokenize(gpt2text_summary))

        metor= [ bert_metor,gpt_metor]
        radardf = pd.DataFrame()
        radardf['ROUGE-1 F1'] = r1
        radardf['ROUGE-2 F1'] = r2
        radardf['ROUGE-L F1 '] = r3
        radardf['BELU'] = belu
        radardf['METOR'] = metor

        fig = go.Figure()
        colors= ["dodgerblue","tomato" , "yellow"]
        for i in range(2):
                fig.add_trace(go.Scatterpolar(r=radardf.loc[i].values, theta=radardf.columns,fill='toself',
                                              name=target[i],
                                              fillcolor=colors[i], line=dict(color=colors[i]),showlegend=True, opacity=0.6))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.0, 0.6])),title="Performance of Models over different evaluation metrics")
        st.write(fig)
        st.table(radardf)


    with tab2:
        st.header("WIKI")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        

          
    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


#     # #st.write(data.columns)
#     #st.write(data[data['paper_id'] == paper_id]['abstract'])

    
   
    


# # https://docs.haystack.deepset.ai/docs/summarizer
# # https://huggingface.co/models?p=1&sort=downloads&search=deepset%2F
# # https://haystack.deepset.ai/tutorials/19_text_to_image_search_pipeline_with_multimodal_retriever
# # https://www.kaggle.com/code/guizmo2000/question-answering-using-cdqa-bert-atos-big-data
#   https://shivanandroy.com/building-a-faster-accurate-covid-search-engine-with-transformers/