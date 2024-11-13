#!/usr/bin/env python
# coding: utf-8

# In[24]:


import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from textblob import TextBlob
from google.cloud import language_v1
import os


# In[35]:


def translate_text(text):
    """Türkçe metni İngilizce’ye çevirir."""
    return GoogleTranslator(source='auto', target='en').translate(text)


# In[37]:


def analyze_sentiment(text):
    """İngilizce metin üzerinde duygu analizi yapar."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity


# In[28]:


def analyze_topic(text):
    """Metnin konusunu analiz eder."""
    
    # JSON dosyanızın yolunu ayarlayın
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/berka/Ses Tahmin Uygulamasi/nimble-orbit-439019-v1-4cdf6e7b609a.json"
    
    client = language_v1.LanguageServiceClient() 
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)

    # Konu analizi yapma
    response = client.analyze_entities(request={'document': document})
    
    entities_info = []
    for entity in response.entities:
        entities_info.append({
            "name": entity.name,
            "type": entity.type_.name,
            "salience": entity.salience
        })
    
    return entities_info


# In[ ]:




