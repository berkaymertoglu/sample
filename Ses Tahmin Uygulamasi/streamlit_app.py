import streamlit as st
from scipy.io import wavfile
import plotly.express as px
import plotly.graph_objects as go
from Real_Time_Predict import ses_kaydi_al, hesapla_mfcc, ses_histogram_goster, tahmin_et, transcribe_audio, kelime_say 
from Emotion_Analysis import translate_text, analyze_sentiment, analyze_topic

def main():
    # Sekmeleri oluştur
    tab1, tab2, tab3 = st.tabs(["Gerçek Zamanlı Ses Tahmini", "Duygu Analizi", "Konu Analizi"])

    with tab1:
        ses_tahmini()  # Gerçek Zamanlı Ses Tahmini sekmesini çağır

    with tab2:
        duygu_analizi()  # Duygu Analizi sekmesini çağır

    with tab3:
        konu_analizi()  # Konu Analizi sekmesini çağır

def ses_tahmini():
    st.title("Ses Analiz Uygulaması")

    if st.button("Ses Kaydı Al"):
        ses_kaydi_al()  # Ses kaydını al
        st.write("Ses kaydı tamamlandı. 'recorded.wav' dosyası kaydedildi.")

        try:
            # MFCC hesapla
            mfccs = hesapla_mfcc('recorded.wav')
            st.write("MFCC Hesaplandı:")
            st.write(mfccs)

            # Ses dosyasını yükle
            sample_rate, audio_time_series = wavfile.read("recorded.wav")

            # Histogramları oluştur ve göster
            fig1, fig2 = ses_histogram_goster(audio_time_series, sample_rate)
            st.pyplot(fig2)  # İlk histogram
            st.pyplot(fig1)  # İkinci histogram

            # Tahmin yap ve olasılıkları al
            predicted_class, probabilities = tahmin_et(mfccs)

            # Olasılıkları yüzde olarak göster
            probabilities_percent = [round(p * 100, 2) for p in probabilities]
            class_labels = ['Berkay', 'Hakan', 'Ekin']

            # Pie chart oluştur
            fig = px.pie(
                values=probabilities_percent,
                names=class_labels,
                title="Kişi Tanıma Olasılık Dağılımı"
            )

            # Pie chart'ı göster
            st.plotly_chart(fig)

            # Tahmin edilen kişiyi göster
            st.write(f"Tahmin Edilen Kişi: {predicted_class}")         

        except Exception as e:
            st.error(f"MFCC hesaplama veya tahmin sırasında hata oluştu: {e}")


def duygu_analizi():
    """Duygu analizi için arayüzü oluşturur."""
    st.title("Duygu Analizi")

    if st.button("Ses Kaydını Analiz Et"):
        # Ses kaydını metne dönüştür
        wav_file_path = "recorded.wav"  # Ses dosyasının yolu
        transcript = transcribe_audio(wav_file_path)

        st.write("Transkript:")
        st.write(transcript)

        # Metindeki kelime sayısını hesapla
        kelimeler = transcript.split()
        st.write("Kelime Sayısı:", len(kelimeler))

        try:
            # Metni İngilizce'ye çevir
            translated_text = translate_text(transcript)
            st.write("İngilizce Çeviri:")
            st.write(translated_text)

            # Duygu analizi yap
            polarity, subjectivity = analyze_sentiment(translated_text)
            st.write(f"Polarity (Duygu Kutupluluğu, Mutluluk): {polarity}")
            st.write(f"Subjectivity (Öznelik): {subjectivity}")

            # Pie chart için verileri hazırla
            labels = ['Mutluluk', 'Üzüntü']
            values = [max(0, polarity), max(0, 1 - abs(polarity))]  # Pozitif ve negatif olasılık

            # Pie chart'ı oluştur
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            st.plotly_chart(fig)  # Pie chart'ı göster

        except Exception as e:
            st.error(f"Duygu analizi sırasında hata oluştu: {e}")
        

def konu_analizi():
    """Konu analizi için arayüzü oluşturur."""
    st.title("Konu Analizi")

    if st.button("Konu Analizi Yap"):
        # Ses kaydını metne dönüştür
        wav_file_path = "recorded.wav"  # Ses dosyasının yolu
        transcript = transcribe_audio(wav_file_path)

        st.write("Transkript:")
        st.write(transcript)

        try:
            # Metni İngilizce'ye çevir
            translated_text = translate_text(transcript)
            st.write("İngilizce Çeviri:")
            st.write(translated_text)

            # Konu analizi yap
            entities_info = analyze_topic(translated_text)
            st.write("Konu Analizi:")
            
            # Pie chart için veri hazırlama
            entity_names = [entity['name'] for entity in entities_info]
            salience_scores = [entity['salience'] for entity in entities_info]

            # Pie chart oluştur
            fig = px.pie(
                values=salience_scores,
                names=entity_names,
                title="Konu Dağılımı"
            )
           
            # Detaylı bilgileri göster
            for entity in entities_info:
                st.write(f"Entity: {entity['name']}, Type: {entity['type']}, Salience: {entity['salience']}")

            # Pie chart'ı göster
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Konu analizi sırasında hata oluştu: {e}")



if __name__ == "__main__":
    main()