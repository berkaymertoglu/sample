#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sounddevice as sd
import wavio  # WAV formatında kaydetmek için

def ses_kaydi_al():
    duration = 5  # Kaydın süresi (saniye)
    sample_rate = 16000  # Örnekleme hızı

    print("Ses kaydı başlıyor...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()  # Kaydın tamamlanmasını bekle
    print("Ses kaydı tamamlandı.")

    wavio.write('recorded.wav', recording, sample_rate, sampwidth=3)
    print("Ses kaydı 'recorded.wav' dosyasına kaydedildi.")


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def ses_histogram_goster(audio_time_series, sample_rate):
    """Sesin spektrogramı ve dalga formu için figürleri oluşturur ve döndürür."""

    # 1. Dalga Formu Figürü
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    time_axis = np.linspace(0, len(audio_time_series) / sample_rate, num=len(audio_time_series))
    ax2.plot(time_axis, audio_time_series, color='green')
    ax2.set_xlabel("Zaman (s)")
    ax2.set_ylabel("Genlik")
    ax2.set_title("Sesin Dalga Formu")
    ax2.grid(True)
    
    # 2. Spektrogram Figürü
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    frequencies, times, Sxx = spectrogram(audio_time_series, sample_rate)
    im = ax1.imshow(
        10 * np.log10(Sxx), aspect='auto', cmap='inferno', origin='lower',
        extent=[times.min(), times.max(), frequencies.min(), frequencies.max()]
    )
    fig1.colorbar(im, ax=ax1, label='Güç Yoğunluğu (dB)')
    ax1.set_xlabel('Zaman (s)')
    ax1.set_ylabel('Frekans (Hz)')
    ax1.set_title('Sesin Zaman-Frekans Spektrumu')
  
    return fig1, fig2  # Figürleri döndürüyoruz


# In[19]:


import librosa
import numpy as np

def hesapla_mfcc(dosya_adi):
    # Ses kaydını yükle
    sample_rate = 44100  # Örnekleme hızı
    y, sr = librosa.load(dosya_adi, sr=sample_rate)

    # MFCC hesapla
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    mfccs = np.mean(mfccs.T, axis=0)  # MFCC'leri ortalayıp tek boyutlu hale getir
    
    return mfccs  # MFCC'leri döndür


# In[21]:


from joblib import load

def tahmin_et(mfccs):

    # Model dosyasının yolunu güncelleyin
    model_path = 'mlp_model.pkl'  # Model dosyasının yolu
    model = load(model_path)  # Modeli yükle

    # MFCC'yi tensöre dönüştür (modelin beklediği şekil)
    mfcc_tensor = np.array(mfccs).reshape(1, -1)  # 1 satır, n sütun şeklinde şekillendirin
    
    # Tahmin edilen sınıf ve olasılıkları al
    probabilities = model.predict_proba(mfcc_tensor)[0]  # Olasılıkları al
    predicted_index = np.argmax(probabilities)  # En yüksek olasılık indisi

    # Sınıf etiketleri
    class_labels = ['Berkay', 'Hakan', 'Ekin']
    
    return class_labels[predicted_index], probabilities


# In[23]:


import speech_recognition as sr

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio, language="tr-TR")
        return transcript
    except sr.UnknownValueError:
        return "Ses anlaşılamadı."
    except sr.RequestError as e:
        return f"Sonuçlar istenemedi; {e}"

def kelime_say(transcript):
    kelimeler = transcript.split()
    return len(kelimeler)  # Kelime sayısını döndür


# In[ ]:




