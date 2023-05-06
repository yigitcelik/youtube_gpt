from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import numpy as np
import random
#yigit

load_dotenv()
os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title='Youtube video Q/A and Summary Bot')

st.header('Youtube video Q/A and Summary Bot')

yt_link = st.text_input(label='Enter the youtube video link that you want to quest:')

# YouTube video URL
if yt_link:
    video_url = yt_link #'https://www.youtube.com/watch?v=oo1ZZlvT2LQ'
    loader = YoutubeLoader.from_youtube_url(video_url)
    try:
        result =  loader.load()
    except:
        loader = YoutubeLoader.from_youtube_url(video_url,language='tr')
        result =  loader.load()



    #summarize the youtube video
    sum_buton = st.button(label='Give Summary info about this video')

    if sum_buton :
        sum_splitter =  RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap = 0)
        sum_texts = sum_splitter.split_documents(result)
        llm_y = OpenAI(temperature = 0)
        chain = load_summarize_chain(llm=llm_y,chain_type='map_reduce',verbose=False)
        sum_result = chain.run(sum_texts)
        st.info(sum_result)


    #Q/A the youtube video
    ask_box = st.text_input(label='Ask your question:')

    if ask_box:
        results=  result[0].page_content

        if '\n' not in results:
            word_list= results.split(' ')
            lenght = len(word_list)
            size= int(lenght/6)
            indice_array = np.arange(100,lenght,size)
            for i in indice_array:
                word_list[i] = word_list[i] + '\n'
            results = ' '.join(word_list)

        splitter = CharacterTextSplitter(separator='\n',chunk_size= 1000,chunk_overlap=200,length_function=len)
        chunks = splitter.split_text(results)
        embeddings = OpenAIEmbeddings()
        thelibrary =FAISS.from_texts(chunks,embedding=embeddings)
        found_doc = thelibrary.similarity_search(ask_box)
        chain = load_qa_chain(OpenAI(temperature = 0), chain_type="stuff")
        st.success(chain.run(input_documents=found_doc, question=ask_box))