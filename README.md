# RAG Financial Intelligence Chatbot

## Proje Hakkında

**RAG Financial Intelligence Chatbot**, yatırımcılara ve finans profesyonellerine gerçek zamanlı piyasa verileri, haber analizleri ve finansal öngörüler sunan gelişmiş bir yapay zeka asistanıdır.

Bu proje, **Retrieval-Augmented Generation (RAG)** mimarisini kullanarak Büyük Dil Modellerini (LLM) güncel finansal verilerle destekler. Böylece standart bir yapay zeka modelinin ötesinde, anlık borsa verilerine, son dakika haberlerine ve şirketlerin detaylı finansal raporlarına dayalı, doğruluk oranı yüksek yanıtlar üretilmesi sağlanır.

Sistem, Ollama altyapısı üzerinden yerel veya bulut tabanlı LLM modelleri (örneğin GPT-OSS, Llama 3) ile çalışacak şekilde tasarlanmıştır.

---

## Temel Özellikler

*   **Gerçek Zamanlı Piyasa Verileri:** Yahoo Finance entegrasyonu ile hisse senedi, kripto para ve endeks verileri anlık olarak temin edilir.
*   **RAG Mimarisi:** Kullanıcı sorguları vektör veritabanında (ChromaDB) taranarak, en alakalı finansal dökümanlarla zenginleştirilmiş yanıtlar oluşturulur.
*   **Haber ve Duygu Analizi:** Piyasa haberleri taranarak yatırımcı duyarlılığı (Sentiment Analysis) ölçülür ve raporlanır.
*   **Teknik ve Temel Analiz:** Fiyat trendleri ve finansal tablolar (Gelir Tablosu, Bilanço, Nakit Akışı) detaylı bir şekilde analiz edilir.
*   **Profesyonel Kullanıcı Arayüzü:** Streamlit tabanlı, modern ve kullanıcı dostu bir web arayüzü sunulmaktadır. Ayrıca Komut Satırı Arayüzü (CLI) desteği de mevcuttur.
*   **Özelleştirilebilir Parametreler:** Sıcaklık (Temperature), Top-K, Top-P gibi model parametreleri arayüz üzerinden dinamik olarak yapılandırılabilir.

---

## Proje Mimarisi

Sistem, modüler bir yapıda tasarlanmış olup şu ana bileşenlerden oluşmaktadır:

1.  **Kullanıcı Arayüzü (UI):** Streamlit (`app.py`) veya CLI (`main.py`) üzerinden kullanıcı ile etkileşim sağlanır.
2.  **Kontrolcü (Controller):** `ChatController`, kullanıcının niyetini (Intent) analiz eder ve ilgili veri kaynaklarını yönetir.
3.  **Veri Toplama (Data Collection):** Yahoo Finance ve çeşitli haber kaynaklarından ham veri toplanır.
4.  **RAG Pipeline:** Elde edilen veriler vektörlere dönüştürülerek ChromaDB'de saklanır ve sorgu esnasında en alakalı bağlam (context) geri çağrılır.
5.  **LLM Entegrasyonu:** Toplanan bağlam, Ollama üzerinden LLM'e iletilir ve nihai yanıt üretilir.

---

## Dosya ve Klasör Yapısı

Aşağıda projenin detaylı dosya yapısı ve açıklamaları yer almaktadır:

```text
financial-chatbot/
├── app.py                      # Web Uygulaması (Streamlit) Başlangıç Dosyası
├── main.py                     # CLI (Komut Satırı) Başlangıç Dosyası
├── config.py                   # Merkezi Konfigürasyon Ayarları
├── requirements.txt            # Proje Bağımlılıkları
├── README.md                   # Proje Dokümantasyonu
├── INSTALL.md                  # Kurulum Rehberi
├── src/                        # Kaynak Kodlar
│   ├── analysis/               # Analiz Modülleri
│   │   ├── market_predictor.py # Piyasa tahmin modelleri
│   │   ├── sentiment.py        # Duygu analizi motoru
│   │   └── trend.py            # Trend analiz araçları
│   ├── chatbot/                # Chatbot Mantığı
│   │   ├── controller.py       # Ana kontrolcü (Orchestrator)
│   │   └── formatter.py        # Yanıt formatlayıcı
│   ├── data/                   # Veri Varlıkları
│   │   └── supported_tickers.py# Desteklenen varlık listesi
│   ├── data_collection/        # Veri Toplama (Scraping)
│   │   ├── news_collector.py   # Haber toplama modülü
│   │   ├── statements_scraper.py # Finansal tablo toplayıcı
│   │   └── yahoo_scraper.py    # Yahoo Finance API arayüzü
│   ├── llm/                    # LLM Bağlantısı
│   │   ├── client.py           # Ollama istemcisi
│   │   └── prompt.py           # Prompt mühendisliği şablonları
│   └── rag/                    # Retrieval-Augmented Generation
│       ├── embeddings.py       # Metin vektörleştirme
│       ├── retriever.py        # Vektör arama ve getirme mantığı
│       └── vector_store.py     # ChromaDB veritabanı yönetimi
└── data/                       # Yerel Veri Depolama (Cache/DB)
```

---

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip ediniz.

### Gereksinimler
*   Python 3.8 ve üzeri
*   [Ollama](https://ollama.com/) (LLM sunucusu için)
*   Git

### Kurulum Adımları

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/seymaduman/financial-chatbot.git
    cd financial-chatbot
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    # Mac/Linux için:
    source venv/bin/activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ollama Modelini Başlatın:**
    Arka planda Ollama servisinin çalıştığından ve konfigürasyondaki modelin (örneğin `gpt-oss:120b-cloud` veya `llama3`) yüklü olduğundan emin olunuz.
    ```bash
    ollama serve
    # Modeli indirmek için (eğer mevcut değilse):
    ollama pull llama3
    ```
    *Not: `config.py` dosyasından kullanmak istediğiniz model adını yapılandırabilirsiniz.*

---

## Kullanım

### 1. Web Arayüzü (Streamlit)
Kullanıcı arayüzünü başlatmak için aşağıdaki komutu çalıştırınız:
```bash
streamlit run app.py
```
Uygulama, varsayılan tarayıcınızda otomatik olarak `http://localhost:8501` adresinde açılacaktır.

**Web Arayüzü Özellikleri:**
*   **Varlık Seçimi (Asset Selection):** Sol menü üzerinden hisse senedi veya kripto para seçimi yapılabilir.
*   **Sohbet Geçmişi (Chat History):** Geçmiş sohbet oturumları görüntülenebilir ve yönetilebilir.
*   **Model Parametreleri (Model Parameters):** Temperature ve Top-K gibi LLM ayarları anlık olarak değiştirilebilir.
*   **Hızlı Analiz (Quick Analysis):** Hazır analiz butonları ile hızlıca rapor alınabilir.
*   **Görselleştirme:** Fiyat grafikleri sohbet akışı içerisinde sunulur.



## Modül Detayları

### `src.chatbot.Controller`
Sistemin merkezi yönetim birimidir. Kullanıcıdan gelen metni işler, `QueryIntent` (niyet) sınıflandırmasını gerçekleştirir (Örneğin: Fiyat bilgisi, Haber talebi, Genel sohbet). Belirlenen niyete göre ilgili `scraper` veya `retriever` modüllerini devreye sokar.

### `src.data_collection`
*   **YahooScraper:** `yfinance` kütüphanesini kullanarak anlık fiyat, hacim ve tarihsel verileri temin eder.
*   **NewsCollector:** İlgili şirket hakkındaki güncel haberleri RSS akışları ve API'ler aracılığıyla toplar.
*   **StatementsScraper:** Şirketlerin gelir tablolarını ve bilançolarını analiz amaçlı olarak getirir.

### `src.rag`
Metin tabanlı verileri (haberler, raporlar) sayısal vektörlere dönüştürür (`embeddings.py`) ve ChromaDB (`vector_store.py`) içerisinde saklar. Kullanıcı sorgusu alındığında, bu veritabanından en yüksek benzerlik skoruna sahip içerikler getirilir (`retriever.py`) ve LLM'e bağlam olarak sunulur.

### `config.py`
Tüm sistem ayarlarının merkezi olarak yönetilmesini sağlar. Ollama sunucu adresi, port bilgisi, embedding modeli ve önbellek (cache) ayarları bu dosyada tanımlanır.

---

## Yasal Uyarı

Bu proje **eğitim ve araştırma amaçlıdır**. Üretilen finansal analizler ve tahminler yatırım tavsiyesi niteliği taşımaz. Yatırım kararları almadan önce lütfen yetkili bir finans danışmanına başvurunuz. Yapay zeka modelleri öngörülemeyen durumlarda hatalı veya eksik bilgi üretebilir.
