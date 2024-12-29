BERT Model Sentiment Analysis

Project Overview

This repository contains scripts and resources for performing sentiment analysis on news articles referring to Russinan-Ukrainian 2022 War using a pre-trained BERT model. The goal is to classify the sentiment of each article as either Pro-Russian or Pro-Ukrainian and calculate a sentiment score.

The scripts were used for my MA thesis

Steps to use the model:

Scraping Journal ariticles from site archives and create a large dataset (Python_Scraper.py script will do the job)

Create a training dataset (for my research I created the narratives.csv)

Create and train the model (BERT_Trainner_Score.py uses Trainer_Dataset.csv as a training dataset)

Evaluate the model finding the deviations of the sentiment score (prediction vs actual) (BERT_Deviation_Finder.py finds and saves the daviations)

Tune the model base on the findings of step 4. (BERT_Tuner.py saves the normalization parameters to a JSON file)

Assess the sentiment of the articles dataset (BERT_Article_Evaluator.py)

Step 1 Python_Scraper.py will scrape Journal Ariticles from URL address. Input your sitemap URL (can be modified to handle multiple sitemaps) and change the name of the CSV to save the download Running Python_Scraper.py will downloading all the articles from URL with the keywords "Russia", "Ukraine", "Zelensky", "Putin" in the Title. All extracted data articles will be stored to a CSV file with 3 Columns: Title,URL,Content

Step 2 Creation of Training Dataset Create a CSV that contain pro-Ukrainian and pro-Russian sentences. The CSV I used is named Trainer_Dataset.csv and contains 3 columns: text,label,score. Score indicates the actual intencity of sentiment on the text in a scale 1-10.

Step 3 BERT_Trainner_Score.py trains a BERT Model with the Trainer_Dataset.csv. The program uses a ration 80% of the dataset for training and 20% for testing. The model is saved in sentiment_model_score.

Step 4 Evaluate Efficiency. BERT_Deviation_Finder.py compares predicted scores with ground truth labels (should be created by the developer) Calculates evaluation metrics to assess model performance. Saves the results in predicted_scores.csv

Step 5 BERT_Tuner.py tunes the trained model by creating a normalization dataframe and saving the parameters to a JSON file (normalization_params.json) for future use. Save this file in sentiment_model_score folder. Now your model is ready to use.

Step 6 BERT_Article_Evaluator_Normalized.py Performs sentiment analysis on a set of articles. It utilizes the dataset from step 2, the model from step 3 and the normalization parameters from step 5. Saves the results with sentiment labels, scores, and corresponding months to a CSV file. (Ensure articles CSVs are structured in 3 Columns: Title,URL,Content)

Results may now be used for your research.
