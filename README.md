
# IMDb Movie Review Sentiment Analysis

An NLP and Machine Learning project that classifies IMDb movie reviews as Positive or Negative using TF-IDF and machine learning algorithms.

## Live Demo

[Open Sentiment Analysis Dashboard](https://sentiment-analysis-dsj8vekxtumvsk57m447sy.streamlit.app/)

## Project Overview

This project uses the IMDb Large Movie Review Dataset containing 50,000 labeled movie reviews.

The objective is to build and evaluate machine learning models that automatically identify the sentiment expressed in movie reviews.

## Models Implemented

- Logistic Regression
- Multinomial Naive Bayes

## Best Model Performance

**Logistic Regression**

| Metric | Score |
|---|---:|
| Accuracy | 89.93% |
| Precision | 89.59% |
| Recall | 90.36% |
| F1-Score | 89.97% |
| ROC-AUC | 96.30% |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- TF-IDF Vectorization

## Project Structure

```text
sentiment-analysis/
├── dashboard/
│   └── app.py
├── images/
├── models/
├── notebooks/
│   └── 01_sentiment_analysis.ipynb
├── reports/
├── requirements.txt
└── README.md
```

## Dataset

IMDb Large Movie Review Dataset (50,000 labeled reviews).

The raw dataset is not included in this repository. It can be obtained from the [IMDb dataset mirror on Hugging Face](https://huggingface.co/datasets/NolanChai/aclImdb_v1).

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/muhammadaqibnoor/sentiment-analysis.git
```

2. Navigate to the project:

```bash
cd sentiment-analysis
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download and extract the IMDb dataset into `data/raw/aclImdb/` if you want to run the notebook.

5. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Disclaimer

Model confidence represents predicted class probability and is not necessarily calibrated. This project is developed for educational and portfolio purposes.

## Author

Muhammad Aqib Noor

[GitHub Profile](https://github.com/muhammadaqibnoor)
