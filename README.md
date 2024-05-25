# SentimentAnalysis_Modelling

# Zaman Serileri Özelinde Derin Öğrenme Modellerinin Karşılaştırılması

Bu proje, Amazon Review veri kümesi üzerinde farklı derin öğrenme modellerinin eğitimini ve sonrasında bu modelin kullanımını içerir.

 **Gereksinimler**
   - Python 3.9
   - nltk
   - NumPy
   - lightgbm
   - textblob
   - wordcloud
   - Matplotlib
   - Scikit-learn

1. **Veri Okuma ve Temizleme**: 
   - Metinleri küçük harfe dönüştürme.
   - Noktalama işaretlerini ve sayıları kaldırma.
   - İngilizce durma kelimelerini kaldırma.
   - Kelime frekanslarına göre düşük frekanslı kelimeleri kaldırma.
   - Kelimeleri köklerine çevirme (lemmatization).

2. **Duygu Analizi**: `SentimentIntensityAnalyzer` sınıfını kullanarak metinlerin duygu analizini yapıyorsunuz. Bu adımda, her bir metnin duygu skorunu hesaplıyorsunuz ve bu skora göre metinlerin pozitif mi yoksa negatif mi olduğunu belirliyorsunuz.

3. **Eğitim-Test Ayırma**: Veri setinizi eğitim ve test alt kümelerine bölmek için `train_test_split` fonksiyonunu kullanıyorsunuz. Bu adımda, 'Title' sütununu veri setinden çıkarıyorsunuz. Eğitim veri seti (`X_train`) içerisinde metinler, test veri seti (`X_test`) içerisinde ise duygu skorları bulunmaktadır. 

4. **Kelime Vektörlerine Dönüştürme**: Metin verilerini sayısal vektörlere dönüştürmek için TF-IDF (Terim Frekansı-Ters Belge Frekansı) vektörleştirmesini kullanıyorsunuz. `world_to_vec` fonksiyonunda, `TfidfVectorizer` sınıfını kullanarak metinleri TF-IDF vektörlerine dönüştürüyorsunuz. Eğitim ve test veri setlerini bu vektörlerle temsil ediyorsunuz.

5. **Model**: LightGBM sınıflandırıcı kullanarak bir model eğitilmiş, ardından bu modelin performansını ROC eğrisi ile görselleştirilmiştir. İlk olarak,  LightGBM sınıflandırıcısını eğitiyor ve ardından bu modelle eğitim ve test veri kümeleri üzerinde tahminler yapıyor. Daha sonra, plot_roc_curve fonksiyonu ile gerçek etiketler ve tahmin edilen olasılıklar arasındaki ROC eğrisini çiziyor. Model sınıflandırma işini %100 gibi bir olasılıkla gerçekleştirmiştir.


