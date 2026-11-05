# Movie Recommendation System (Demo 0.0.1)

## 1. Introduction

This project develops a Movie Recommendation System using Machine Learning techniques and MovieLens datasets.

## 2. Datasets

This project uses two datasets from GroupLens:

### MovieLens 100K
- Download: https://grouplens.org/datasets/movielens/100k/
- Description: MovieLens 100K movie ratings. Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.

### MovieLens 1M
- Download: https://grouplens.org/datasets/movielens/1m/
- Description: MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.

## 3. Dataset Installation

Download the datasets using the links above and extract them into the `data/` directory.

Expected folder structure:

```text
data/
├── ml-100k/
└── ml-1m/
```

**Note:** The datasets are not included in this GitHub repository. Users must download them separately.

## 4. Installation

```bash
pip install -r requirements.txt
```

## 5. Run Application

```bash
streamlit run app/main.py
```
