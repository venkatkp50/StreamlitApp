import io
import os
import re
import json
import time
import math
import spacy
from spacy import displacy
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
from nltk.corpus import stopwords
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
nltk.download('stopwords')
new_stopwords = ["What"]
BERT_MAX_TOKEN = 512
GPT2_MAX_TOKEN = 1024
import warnings
warnings.filterwarnings('ignore')

# Stopword = stopwords.words('english') 
# Stopword.extend(new_stopwords)
# NER = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

imagename2 = Image.open('images/Sidebar2.jpg')
st.sidebar.image(imagename2)
st.sidebar.title('Settings')
modelSelected = st.sidebar.selectbox('Choose Reader Model',options=('deepset/roberta-base-squad2-covid','deepset/roberta-base-squad2','deepset/covid_bert_base'))
imagename = Image.open('images/caronavirus banner.jpg')
st.image(imagename)
st.text_input("Your Query", key="input_text",value='')
load = st.button('Search')

# punk = string.punctuation
# user_message = st.session_state.input_text
# textKeywords = user_message
# nerKewword= NER(user_message)
# # nerlist = nerKewword.ents
# nerlist= [itm for itm in nerlist]
# textlist = [word for word in textKeywords.split() if word not in Stopword if word not in punk]
# for word in nerlist:
#   if str(word) in textlist:
#     textlist.remove(str(word))
# print(textlist)

all_files = []
json_filepath = 'json_files'
for dirname in os.listdir(json_filepath):
    filenames = os.listdir(json_filepath + '/' + dirname )
    for filename in filenames:
        file = json.load(open(json_filepath + '/' + dirname  + '/' + filename, 'rb'))
        all_files.append(file)

# st.write('...1')

file = all_files[0]
# st.write("Dictionary keys:", file.keys())
# st.write('...2')   

def format_name(author):
    middle_name = " ".join(author['middle'])    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])

#st.write('...3')

def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

#st.write('...4')

def format_authors(authors, with_affiliation=False):
    name_ls = []    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)    
    return ", ".join(name_ls)

#st.write('...5')

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}    
    for section, text in texts:
        texts_di[section] += text
    body = ""
    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"    
    return body

#st.write('...6')

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))
    return "; ".join(formatted)

#st.write('...7')

cleaned_files = []
for file in (all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
        format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    cleaned_files.append(features)

#st.write('...8')

col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]
data = pd.DataFrame(cleaned_files, columns=col_names)

text_file_path = 'text_file'
abstract_file_path = 'abstract_file'
bert_file_summary_path = 'summary_file/BERT'
gpt_file_summary_path = 'summary_file/GPT'

if len(os.listdir(text_file_path)) < 5:
    #st.write('Create text files for Haystack')
    txtfile = {}
    for i in range(data.shape[0]):
        txtfile[data.loc[i,'paper_id']] = data.loc[i,'text']
    for key,val in txtfile.items():
        with open(text_file_path+'/'+key+'.txt', 'w') as f:
            f.write(str(val))

if len(os.listdir(abstract_file_path)) < 5:
    #st.write('Create abstract files for Haystack')
    abstractfile = {}
    for i in range(data.shape[0]):
        abstractfile[data.loc[i,'paper_id']] = data.loc[i,'abstract']
    #st.write('Creatint abstract files ......')
    for key,val in abstractfile.items():
        with open(abstract_file_path+'/'+key+'.txt', 'w') as f:
            f.write(str(val))
    #st.write('Creatint abstract files ......completed')

doc_dir = text_file_path

document_store = InMemoryDocumentStore(use_bm25=True)
docs = convert_files_to_docs(dir_path=doc_dir,clean_func=clean_wiki_text,split_paragraphs=True)
document_store.write_documents(docs)
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path=modelSelected, use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

#st.write('.....9')

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
    ans = responsedf['Probable Anwsers'].values.tolist()
    ids = responsedf['Source File Name'].values.tolist()
    scorelist = responsedf['Score'].values.tolist()
    scorelist = [ x*100 for x in scorelist]

    responsedf = responsedf.astype(str).apply(lambda x: x.str[:30])
    ansfig = responsedf['Probable Anwsers'].values.tolist()
    #st.table(responsedf[['Probable Anwsers','Score']])
    
    max_score = float(responsedf['Score'].max())
    if max_score >  0.9:
        scoremultiplier = 90        
    elif max_score > 0.7:
            scoremultiplier = 150
    elif max_score > 0.4:
            scoremultiplier = 175
    else:
            scoremultiplier = 200

    score100 = [scr*scoremultiplier for scr in score]
    
    #colorcode = ['rgb(116, 191, 0)', 'rgb(60, 194, 0)', 'rgb(2, 198, 0)', 'rgb(0, 210, 186)', 'rgb(0, 174, 213)']
    colorcode = ['rgb(102, 0, 51)', 'rgb(204, 0, 102)', 'rgb(255, 51, 153)', 'rgb(102, 255, 255)', 'rgb(204, 204, 255)']
    opacitycode = [0.8, 0.6, 0.5, 0.4,0.3]
    fig = go.Figure(data=[go.Scatter(x=ansfig, y=scorelist,marker=dict(color=colorcode,opacity=opacitycode,size=score100,))])
    st.subheader('Responses..')
    st.markdown('----')
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

    
    col1.write(ans[0])
    col2.write(ans[1])
    col3.write(ans[2])
    col4.write(ans[3])
    col5.write(ans[4])
    st.markdown('--')
    col1.write(str(round(score[0],2)*100)+'%')
    col2.write(str(round(score[1],2)*100)+'%')
    col3.write(str(round(score[2],2)*100)+'%')
    col4.write(str(round(score[3],2)*100)+'%')
    col5.write(str(round(score[4],2)*100)+'%')
    st.markdown('---')
    st.subheader('Score %')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True,)
    
    selected_radio = st.radio('Choose File for Summarization',options=(ans[0],ans[1],ans[2],ans[3],ans[4]))

    file4Summ = ''
    filecount = 0
    #file4Summ = id[0]
    if selected_radio == ans[0]:
        filecount = 0
    elif selected_radio == ans[1]:
        filecount = 1
    elif selected_radio == ans[2]:
        filecount = 2
    elif selected_radio == ans[3]:
        filecount = 3
    else:
        filecount = 4
    
    
    def getTextSummarization(filecount,summarizationFor,std_text,tot_words_ref):
        if summarizationFor == 'std':
            text = open(abstract_file_path+'/'+id[filecount],"r").readlines()
            text = ' '.join(sent for sent in text if len(sent.strip()) != 0 )
            return text
        if summarizationFor == 'BERT':
            filenames = os.listdir(bert_file_summary_path)
            if id[filecount] in filenames:
                text = open(bert_file_summary_path+'/'+id[filecount],"r").readlines()
                text = ' '.join(sent for sent in text)
                return text
            else:
                header =[]
                berttext = []
                para = []
                bert_model = Summarizer() 

                if tot_words_ref > BERT_MAX_TOKEN:

                    for line in std_text:
                        if len(line) > 1:
                            if len(line) < 100:
                                header.append(line)
                            else:
                                para.append(line)                  
                
                    for parabody in para:
                        berttext.append(bert_model(body=parabody,max_length=100))
                    
                    berttext = bert_model(body=parabody,max_length=math.ceil(tot_words_ref / 100) * 100)
                    bert_summary = ''.join( lines for lines in berttext) 
                else:
                    for line in std_text:
                        para.append(line) 
                    berttext = ''.join( lines for lines in para) 
                    berttext = bert_model(body=berttext,max_length=math.ceil(tot_words_ref / 100) * 100)
                    bert_summary = ''.join( lines for lines in berttext) 

                with open(bert_file_summary_path+'/'+id[filecount], 'w') as createFile:
                    createFile.write(bert_summary)               
            return bert_summary
        
        if summarizationFor == 'GPT2':            
            filenames = os.listdir(gpt_file_summary_path)
            if id[filecount] in filenames:
                text = open(gpt_file_summary_path+'/'+id[filecount],"r").readlines()
                text = ' '.join(sent for sent in text)
                return text
            else:
                header =[]
                para = []
                gpt2text = []
                GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")

                if tot_words_ref > GPT2_MAX_TOKEN:
                    
                    for line in std_text:
                        if len(line) > 1:
                            if len(line) < 100:
                                header.append(line)
                            else:
                                para.append(line)                  
                    for parabody in para:
                        gpt2text.append(GPT2_model(body=parabody, max_length=100))
                
                    gpt2text_full = ''.join(text for text in gpt2text)
                    gpt2text_full = GPT2_model(body=gpt2text_full, max_length=math.ceil(tot_words_ref / 100) * 100)
                else:
                    for line in std_text:
                        para.append(line) 

                    gpt2text = ''.join( lines for lines in para) 
                    gpt2text = GPT2_model(body=gpt2text,max_length=math.ceil(tot_words_ref / 100) * 100)
                    gpt2text_full = ''.join( lines for lines in gpt2text) 

                with open(gpt_file_summary_path+'/'+id[filecount], 'w') as createFile:
                    createFile.write(gpt2text_full)               
            return gpt2text_full


    tab1, tab2 = st.tabs(["Single Document Summarization", "Multi Document Summarization"])
    with tab1:
        col1 , col2 , col3 = st.columns([1,1,1])
        col1.write('Reference Standard')
        gold_text = getTextSummarization(filecount,'std','',0)
 
        while (len(gold_text.strip()) == 0 and filecount < 5):
            filecount = filecount+1
            gold_text = getTextSummarization(filecount+1,'std','',0)
        print('STD filecount=',filecount)
        tot_words_ref = len(word_tokenize(gold_text))
        col1.write(gold_text)  

        col2.write('BERT Summarization')
        print('BERT filecount=',filecount)
        full_text = open(text_file_path+'/'+id[filecount],"r").readlines()
        bert_summary = getTextSummarization(filecount,'BERT',full_text,tot_words_ref)
        col2.write('Abstract : This article describes,' + bert_summary)  

        col3.write('GPT-2 Summarization')
        print('GPT filecount=',filecount)
        gpt2text_summary = getTextSummarization(filecount,'GPT2',full_text,tot_words_ref)
        col3.write('Abstract : This article describes,' + gpt2text_summary )  
        st.markdown('----')
        st.subheader('Summarization Statistics')

        col1 , col2, col3 = st.columns(3)

        tot_words_bert = len((bert_summary.split()))
        tot_words_gpt3 = len((gpt2text_summary.split()))
        col1.metric('Total Words Reference Text',tot_words_ref)
        col2.metric("Total Words BERT Summarization", tot_words_bert,(tot_words_bert - tot_words_ref))
        col3.metric("Total Words GPT-3 Summarization",tot_words_gpt3,(tot_words_gpt3 - tot_words_ref) )

        tot_words_ref = len(sent_tokenize(gold_text))
        tot_words_bert = len(sent_tokenize(bert_summary))
        tot_words_gpt3 = len(sent_tokenize(gpt2text_summary))
        
        col1.metric('Total Sentences Reference Text',tot_words_ref)
        col2.metric("Sentences in BERT Summarization", tot_words_bert,(tot_words_bert - tot_words_ref))
        col3.metric(" Sentences in GPT-2 Summarization",tot_words_gpt3,(tot_words_gpt3 - tot_words_ref) )
        st.markdown('----')

        st.subheader('Performance Analysis of Text-Summary')     
        rouge = rouge.Rouge()
        bertscores = rouge.get_scores(hyps=gold_text, refs=bert_summary, avg=True)        
        gpt2scores = rouge.get_scores(hyps=gold_text, refs=gpt2text_summary, avg=True)   

        col1, col2, col3 = st.columns(3)
        
        col2.write('BERT Score')
        bertscore = pd.DataFrame(bertscores)
        col2.table(bertscore)

        col3.write('GPT-2 Score')
        gpt2score = pd.DataFrame(gpt2scores)
        col3.table(gpt2score)

        dfbert = bertscore.T
        dfbert['Model'] = 'BERT'
        dfgpt = gpt2score.T
        dfgpt['Model'] = 'GPT-2'
        df = pd.concat([dfbert,dfgpt])

        st.markdown('---')

        target  = ['BERT','GPT-2']
        r1 = [bertscore.loc['f','rouge-1'],gpt2score.loc['f','rouge-1']]
        r2 = [bertscore.loc['f','rouge-2'],gpt2score.loc['f','rouge-2']]
        r3 = [bertscore.loc['f','rouge-l'],gpt2score.loc['f','rouge-l']]
        refs = [sent.split() for sent in sent_tokenize(gold_text)]
        cands  = [ cand for cand in bert_summary.split()]
        bert_beluscore = sentence_bleu(refs, cands)
        cands  = [cand for cand in gpt2text_summary.split()]
        gpt_beluscore = sentence_bleu(refs, cands)
        belu = [ bert_beluscore,gpt_beluscore]

        bert_metor = meteor([word_tokenize(gold_text)],word_tokenize(bert_summary))
        gpt_metor = meteor([word_tokenize(gold_text)],word_tokenize(gpt2text_summary))

        metor= [ bert_metor,gpt_metor]
        radardf = pd.DataFrame()
        radardf['ROUGE-1 F1'] = r1
        radardf['ROUGE-2 F1'] = r2
        radardf['ROUGE-L F1 '] = r3
        radardf['BELU'] = belu
        radardf['METOR'] = metor

        fig = go.Figure()
        colors= ["dodgerblue", "yellow", "tomato" ]
        for i in range(2):
                fig.add_trace(go.Scatterpolar(r=radardf.loc[i].values, theta=radardf.columns,fill='toself',
                                              name=target[i],
                                              fillcolor=colors[i], line=dict(color=colors[i]),showlegend=True, opacity=0.6))
        st.subheader("Performance of Models over different evaluation metrics")
        radarmax = radardf.max()
        radmaxval =  radarmax.max()             
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.0, radmaxval])),)
        st.write(fig)
        st.table(radardf)


    
    with tab2:
        st.subheader("Multi Document Summarization")



# #st.write('.....10')