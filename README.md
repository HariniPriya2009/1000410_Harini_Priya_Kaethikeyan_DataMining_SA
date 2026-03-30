# 1000410_Harini_Priya_Kaethikeyan_DataMining_SA

 Data Driven Black Friday Sales Insights

This project applies data mining techniques to analyze Black Friday retail sales data and extract actionable business insights.

📊 Project Overview

This project analyzes the Black Friday sales dataset to understand customer shopping behavior, segment customers, discover product associations, and detect anomalies. The insights help retailers optimize marketing strategies and improve customer engagement.

Steamlit Link:  

🎯 Project Objectives

Identify Shopping Behaviors - Understand customer purchase patterns

Customer Segmentation - Group customers into meaningful clusters

Product Associations - Find products frequently bought together

Anomaly Detection - Identify unusual high spenders

Deploy Insights - Build an interactive Streamlit dashboard

Data Mining Steps

1. Data Loading

Load BlackFriday_Cleaned.csv using pandas
Dataset contains 537,577 transactions with 15 features

2. Data Cleaning & Preprocessing

Handle missing values in Product_Category_2 and Product_Category_3

Encode categorical variables (Gender, Age, City_Category)

Remove duplicates

Normalize Purchase using StandardScaler

3. Exploratory Data Analysis (EDA)

Histogram of Purchase distribution

Boxplot of Purchase by Gender

Countplot of Product Categories

Scatter plot of Age vs Purchase

Correlation Heatmap

4. Clustering Analysis

Applied K-Means clustering

Features used: Age, Occupation, Purchase

Used Elbow Method to determine optimal clusters (K=3)

Identified three customer segments:

Budget Shoppers: Low purchase amounts

Regular Shoppers: Moderate purchase amounts

Premium Buyers: High purchase amounts

5. Association Rule Mining

Used Apriori algorithm from mlxtend

Generated frequent itemsets from product categories

Created association rules with support, confidence, and lift metrics

Identified product combinations frequently bought together

6. Anomaly Detection

Applied Z-score method to detect outliers

Identified high spenders (transactions above 3 standard deviations)

Analyzed anomaly distribution by age and gender

📈 Key Insights

Purchase Statistics

Average Purchase: $9,333.86

Median Purchase: $8,062.00

Total Revenue: $5.02 Billion

Top Product Categories

Category 5 - Most popular product category

Category 1 - Second most popular

Category 8 - Third most popular

Customer Segments

Segment	Count	Avg Purchase

Budget Shoppers	~30%	~$6,000

Regular Shoppers	~40%	~$9,500

Premium Buyers	~30%	~$14,000

Anomalies Detected

Total anomalies: 2,665 (0.5% of transactions)

Average anomaly purchase: ~$22,000

Most anomalies from age groups 26-35 and 36-45

🎯 Clustering Results

The K-Means algorithm identified three distinct customer segments:

Budget Shoppers: Customers with lower purchase amounts, tend to buy essentials

Regular Shoppers: Average spending customers, potential for upselling

Premium Buyers: High-value customers, ideal for loyalty programs

🔗 Association Rules

Key product associations discovered:

Products in Category 5 are frequently bought with Category 8

Category 1 products often co-occur with Category 2 products

Cross-selling opportunities identified for product bundling

⚠️ Anomaly Detection

Used Z-score method with threshold of 3

Detected 2,665 high-spending transactions

These customers could be targeted for VIP programs

🖥️ Streamlit Deployment

The project includes an interactive Streamlit dashboard with the following features:

Dataset Overview: View dataset statistics and preview

EDA: Interactive visualizations for data exploration

Clustering: Adjustable cluster count with visual results

Association Rules: Configurable support and confidence thresholds

Anomaly Detection: Adjustable Z-score threshold

Insights: Key findings and recommendations

Deploy on Streamlit Cloud:

Push code to GitHub repository

Go to share.streamlit.io

Connect your GitHub repository

Deploy the app.py file

📊 Technologies Used

Python 3.11 - Programming language

Pandas - Data manipulation

NumPy - Numerical operations

Matplotlib - Data visualization

Seaborn - Statistical visualization

Scikit-learn - Machine learning (K-Means, StandardScaler)

Mlxtend - Association rule mining (Apriori)

SciPy - Statistical analysis (Z-score)

Streamlit - Web dashboard



