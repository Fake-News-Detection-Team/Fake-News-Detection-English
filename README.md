# 📰 Fake News Detection System — English

### Hybrid CNN-LSTM Based Fake News Classification using Natural Language Processing

A deep learning-based **Fake News Detection System** developed to classify English news content as **Real** or **Fake**.

The system uses a **Hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)** architecture to extract meaningful textual features and learn sequential relationships within news content.

A **Flask web application** provides a simple interface through which users can enter news content and receive a prediction with a confidence score.

---

## 📌 Overview

The rapid growth of digital media, online news platforms, and social networking has made the spread of misleading and fabricated information increasingly difficult to control.

This project aims to address this problem by applying **Natural Language Processing (NLP)** and **Deep Learning** techniques to automatically analyze textual news content and classify it as either **Real News** or **Fake News**.

The system processes the input text, converts it into a numerical representation using a trained tokenizer, and passes it through the trained Hybrid CNN-LSTM model for classification.

### 🎯 Main Objective

> To develop an automated deep learning system capable of identifying potentially fake English news based on its textual content.

---

## ✨ Features

* 🔎 **English Fake News Detection**
* 🧠 **Hybrid CNN-LSTM Deep Learning Model**
* 🌐 **Flask Web Application**
* 📝 News headline and text input
* 📊 Prediction confidence score
* 🧹 Automated text preprocessing
* 🔤 Text tokenization and sequence processing
* 🤖 Pre-trained Keras model for prediction
* 💻 Simple and user-friendly interface
* 📦 Saved tokenizer for consistent inference
* ⚡ Fast classification using the trained model

---

## 🧠 System Architecture

The overall workflow of the system is:

```text
                    ┌─────────────────────┐
                    │      User Input     │
                    │  Headline + News    │
                    │       Content       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Text Preprocessing │
                    │                     │
                    │ • Lowercasing       │
                    │ • URL Removal       │
                    │ • HTML Removal      │
                    │ • Punctuation      │
                    │ • Stopwords        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Tokenization    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Sequence Padding   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        CNN          │
                    │ Feature Extraction  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        LSTM         │
                    │ Sequential Learning │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Classification Layer│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Prediction     │
                    │                     │
                    │   REAL / FAKE       │
                    │   + Confidence      │
                    └─────────────────────┘
```

---

## 🔬 Hybrid CNN-LSTM Model

The core of the project is a **Hybrid CNN-LSTM neural network**.

### Convolutional Neural Network (CNN)

The CNN component extracts important local features from the processed textual representation.

It helps identify patterns such as:

* Important word combinations
* Phrase-level features
* Local textual patterns
* Discriminative features within news content

### Long Short-Term Memory (LSTM)

The LSTM component processes the extracted features sequentially and learns relationships between different parts of the text.

This allows the model to capture:

* Sequential dependencies
* Contextual relationships
* Long-term textual patterns
* Relationships between different parts of the news content

### Final Classification

The learned representation is passed to the classification layer, which produces the final prediction.

The system classifies the news into:

```text
REAL
```

or

```text
FAKE
```

The Flask application then presents the result along with the model's confidence score.

---

## 🔄 NLP Preprocessing Pipeline

Before the news content is supplied to the neural network, it undergoes several preprocessing operations.

### 1. Text Cleaning

Unnecessary elements such as URLs, HTML tags, punctuation, and unwanted characters are removed.

### 2. Lowercasing

The text is converted into lowercase to maintain consistency.

### 3. Stopword Processing

Common words that provide limited classification value can be removed during preprocessing.

### 4. Tokenization

The trained tokenizer converts words into numerical token IDs that can be processed by the neural network.

Example:

```text
"Government announces new policy"
```

is converted into a numerical sequence representing the corresponding tokens.

### 5. Sequence Padding

The tokenized sequence is padded to the required input length so that it can be passed to the trained model.

### 6. Model Prediction

The processed sequence is passed through the Hybrid CNN-LSTM model to obtain the final classification.

---

## 📊 Model Performance

The English version of the system achieved approximately the following results during evaluation:

| Metric       |      Result |
| ------------ | ----------: |
| 🎯 Accuracy  |  **99.66%** |
| 🎯 Precision |  **99.72%** |
| 🎯 Recall    |  **99.63%** |
| 🎯 F1-Score  | **~99.67%** |

These results indicate strong classification performance on the evaluated English fake-news dataset.

> **Note:** Model performance can vary depending on the dataset, preprocessing pipeline, train/test split, and evaluation methodology.

---

## 📚 Dataset

The project was developed using publicly available English fake-news datasets, including the **ISOT Fake News Dataset** and related English news data used during development and evaluation.

The dataset contains examples belonging to two primary classes:

* **Real News**
* **Fake News**

The textual information is processed through the NLP pipeline before being supplied to the deep learning model.

---

## 🧪 Model Evaluation

Several standard classification metrics can be used to evaluate the performance of the system.

### Accuracy

Measures the percentage of correctly classified news samples.

### Precision

Measures how many of the samples predicted as a particular class actually belong to that class.

### Recall

Measures how effectively the model identifies samples belonging to a particular class.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

A confusion matrix can be used to analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC-AUC

ROC curves and Area Under the Curve (AUC) can be used to further evaluate the classification capability of the model across different decision thresholds.

---

## 🌐 Web Application

The trained model is integrated into a **Flask-based web application**.

The application provides a simple interface where users can enter English news content and request a prediction.

### User Workflow

```text
1. Enter News Headline
          ↓
2. Enter News Content
          ↓
3. Click Check / Predict
          ↓
4. Text Preprocessing
          ↓
5. CNN-LSTM Model Prediction
          ↓
6. Display Result
          ↓
7. Display Confidence Score
```

### Example Output

```text
Prediction: REAL

Confidence: 98.XX%
```

or

```text
Prediction: FAKE

Confidence: 97.XX%
```

---

## 📂 Project Structure

```text
Fake-News-Detection-English/
│
├── app.py
│
├── models/
│   └── fake_news_detector_English.keras
│
├── tokenizers/
│   └── tokenizer_English.pickle
│
├── static/
│   ├── logo.png
│   └── background.png
│
├── templates/
│   └── index.html
│
├── English News.docx
│
├── .gitignore
│
└── README.md
```

### 📄 File Description

| File / Folder       | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| `app.py`            | Main Flask application and prediction logic                   |
| `models/`           | Contains the trained Keras CNN-LSTM model                     |
| `tokenizers/`       | Contains the trained tokenizer                                |
| `static/`           | Static assets such as images and interface resources          |
| `templates/`        | HTML templates used by the Flask application                  |
| `English News.docx` | English news material used during project development/testing |
| `.gitignore`        | Specifies files that should not be tracked by Git             |
| `README.md`         | Project documentation                                         |

---

## 🛠️ Technologies Used

### Programming

* 🐍 Python

### Deep Learning

* TensorFlow
* Keras
* CNN
* LSTM

### Natural Language Processing

* NLTK
* Text preprocessing
* Tokenization
* Sequence processing

### Data Science / Machine Learning

* NumPy
* Pandas
* Scikit-learn

### Web Development

* Flask
* HTML
* CSS
* Jinja2

### Development Tools

* Jupyter Notebook
* Visual Studio Code

### Version Control

* Git
* GitHub

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Yash-Agrawal-2004/Fake-News-Detection-English.git
```

### 2. Navigate to the Project Directory

```bash
cd Fake-News-Detection-English
```

### 3. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

For Linux/macOS:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

### 4. Install Required Packages

```bash
pip install flask tensorflow keras numpy pandas nltk scikit-learn
```

If a `requirements.txt` file is provided in the repository, install dependencies using:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

The Flask development server will start.

Open the local address displayed in the terminal, usually:

```text
http://127.0.0.1:5000/
```

---

## 🖥️ Application Screenshots

### 🏠 Application Interface

The system provides a simple and user-friendly interface where users can select the English language and enter a news headline and article for analysis.

![English Fake News Detection System - Home Page](screenshots/home.png)

---

### ❌ Fake News Prediction

The system processes the submitted news article using the trained Hybrid CNN-LSTM model and displays the predicted class along with its confidence score.

![Fake News Prediction](screenshots/fake-news-prediction.png)

---

### ✅ Real News Prediction

The application can also identify news content classified as real and display the corresponding confidence score.

![Real News Prediction](screenshots/real-news-prediction.png)

---

## 🔐 Disclaimer

This project is intended for **academic, educational, and research purposes**.

The prediction generated by this system should **not be considered definitive proof** that a news article is true or false.

Machine learning models can produce incorrect predictions, especially when presented with:

* Unfamiliar topics
* New writing styles
* Information outside the training dataset
* Satirical content
* Highly specialized subjects
* Manipulated or ambiguous information

Important news and claims should always be verified using reliable and authoritative sources.

---

## 🔮 Future Enhancements

The system can be further improved by incorporating:

* 🌍 Multilingual fake-news detection
* 📰 News source credibility analysis
* 🔗 News URL-based prediction
* 🤖 Transformer-based models such as BERT
* 📱 Responsive mobile interface
* 📊 Advanced analytics dashboard
* 🔍 Explainable AI for prediction interpretation
* ☁️ Cloud deployment
* 🔄 Continuous model retraining
* 🧪 Larger and more diverse datasets
* 🕵️ Detection of AI-generated or manipulated news content

---

## 🎓 Academic Contribution

This project demonstrates the application of **Natural Language Processing and Deep Learning** to a real-world information reliability problem.

The project combines:

```text
Natural Language Processing
          +
Text Preprocessing
          +
CNN Feature Extraction
          +
LSTM Sequential Learning
          ↓
English Fake News Classification
```

It provides practical experience in:

* NLP
* Deep Learning
* Text Classification
* CNN-LSTM Architecture
* Model Evaluation
* Flask Application Development
* Model Deployment and Integration
* Git and GitHub

---

## 👨‍💻 Authors

### Yash Agrawal

**B.Tech — Information Technology**

GitHub: **@Yash-Agrawal-2004**

### Sarthak Srivastava

**B.Tech — Information Technology**

GitHub: **@Satty36-s**

---

## ⭐ Acknowledgements

This project makes use of open-source technologies and publicly available datasets.

Special thanks to the developers and research communities behind:

* TensorFlow
* Keras
* Flask
* NLTK
* Scikit-learn
* Publicly available fake-news datasets

---

## 📜 License

This project is primarily intended for **academic and educational purposes**.

Before redistributing or commercially using this project, ensure that the licensing requirements of all third-party libraries, datasets, images, and other resources used by the project are respected.

---

## ⭐ If You Find This Project Useful

If this project is useful for learning or research, consider giving the repository a ⭐ on GitHub.
