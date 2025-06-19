# ShopBoys Customer Purchase Propensity Predictor

## Project Overview

This project addresses a key challenge faced by ShopSavvy, an online retailer: effectively identifying which user sessions are most likely to result in a purchase. By predicting purchase intent, ShopSavvy can optimize its marketing efforts, personalize the user experience in real-time, and enable proactive interventions by sales or customer support teams, ultimately driving increased sales and improving marketing ROI.

The specific business objective guiding this project, as prioritized by the Marketing and Product teams, is to **maximize the identification of potential purchasers**. This translates to building a model with high **Recall** for the 'Purchase' class, ensuring that as few genuine purchase opportunities are missed as possible, even if it means a slightly higher rate of false positives compared to a model prioritizing precision.

## Business Problem

ShopSavvy was facing inefficiencies due to a lack of insight into immediate user purchase likelihood. This resulted in:

*   Wasted marketing budget on users with low purchase intent.
*   Generic user experiences that didn't capitalize on high-potential sessions.
*   Missed opportunities to engage with users who were close to converting.

A predictive model is needed to transform raw session data into actionable intelligence about user intent.

## Project Goal & Research Question

The primary goal is to develop a robust and deployable machine learning model that accurately predicts the likelihood of a user session resulting in a purchase.

The core research question explored is:

> What online shopper behavioral and technical features best predict purchase intention, and how can a predictive model leverage these features to support ShopSavvy's goal of maximizing sales capture?

## Dataset

The analysis is based on the publicly available [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) from the UCI Machine Learning Repository. This dataset mirrors the type of session data collected by ShopSavvy, containing 12,330 user sessions with 18 features describing page visits, engagement metrics, technical attributes, and session time. The target variable is `Revenue`, indicating whether a purchase occurred.

## Methodology

The project followed a structured data science workflow:

1.  **Data Loading and Initial Exploration:** Loaded the dataset and performed initial checks to understand its structure and identify the target variable.
2.  **Exploratory Data Analysis (EDA):** Conducted detailed analysis and visualizations to understand feature distributions, identify relationships between features, and uncover how different variables correlate with purchase intent. This phase was crucial for identifying potential drivers of revenue and informing feature selection.
3.  **Feature Importance Analysis:** Used a tree-based model to quantify the predictive power of each feature.
4.  **Feature Engineering - Behavioral Clustering:** Applied K-Means clustering to identify distinct user session behaviors and added the cluster assignment as a new feature. This aimed to capture complex interaction patterns that might be predictive of purchase.
5.  **Model Selection, Training, and Evaluation:**
    *   Prepared data for modeling, including encoding categorical features and incorporating the engineered cluster feature.
    *   Addressed the class imbalance in the target variable using Random Over-sampling on the training data to improve the model's ability to predict the minority 'Purchase' class.
    *   Selected LightGBM and Random Forest, powerful classification algorithms suitable for this problem.
    *   Performed Hyperparameter Tuning using Randomized Search with cross-validation to optimize model performance, specifically focusing on metrics relevant to the imbalanced class.
    *   Trained the final, tuned models and evaluated their performance on an unseen test set using key metrics like Precision, Recall, F1-score, and ROC AUC, with a focus on the 'Purchase' class.
6.  **Model Deployment Preparation:** Prepared the final selected model within a robust pipeline that includes necessary preprocessing steps, ready for deployment.

## Key Findings & Actionable Insights

Through EDA, feature importance, and clustering, several key factors strongly associated with purchase intent were identified:

*   **High Engagement:** Sessions with a higher number of page views and longer durations across Administrative, Informational, and Product-Related pages are significantly more likely to convert.
    *   *Insight:* Encourage users to explore more content through clear navigation and internal linking.
*   **Page Value:** The `PageValues` feature, representing the value of pages visited, shows the strongest positive correlation with `Revenue`.
    *   *Insight:* Identify pages with high `PageValues` and optimize them for conversion. Drive traffic to these high-value pages.
*   **Low Bounce and Exit Rates:** Sessions where users leave the site quickly or from entrance pages are less likely to result in a purchase.
    *   *Insight:* Improve landing page quality, optimize content relevance, and ensure smooth navigation to reduce immediate exits.
*   **Visitor Type & Clusters:** 'New_Visitor' and 'Other' visitor types show a higher conversion *rate*, though 'Returning_Visitor' contribute more total purchases. Clustering revealed distinct behavioral segments; Cluster 1 (Highly Engaged Returning Visitors) had significantly higher conversion rates.
    *   *Insight:* Develop tailored strategies for different visitor types and behavioral clusters. For Cluster 1 users, focus on reinforcing intent (e.g., abandoned cart reminders, loyalty offers). For Cluster 2 (Low Engagement), focus on initial engagement and guidance.
*   **Seasonality & Timing:** Certain months (like November, December) and weekdays showed higher proportions of revenue-generating sessions.
    *   *Insight:* Plan marketing campaigns and promotions to align with peak purchase periods.

## Model Selection & Evaluation Results

After exploring both LightGBM and Random Forest and tuning their hyperparameters on resampled training data, we evaluated their performance on the original imbalanced test set. Given ShopSavvy's priority to **maximize the identification of potential purchasers (Recall for the 'True' class)**, we focused on the metrics reflecting the model's ability to capture true positive cases.

Here's a summary of the performance of the tuned models on the test set:

| Metric            | Tuned LightGBM | Tuned Random Forest |
| :---------------- | :------------- | :------------------ |
| Precision (True)  | 0.68           | **0.75**            |
| Recall (True)     | **0.63**       | 0.55                |
| F1-Score (True)   | 0.65           | 0.64                |
| ROC-AUC Score     | 0.913          | **0.919**           |

*(Note: Metrics may vary slightly based on exact tuning process and random seeds)*

Based on these results, the **Tuned LightGBM** model demonstrated the highest **Recall** for the 'True' (Purchase) class (0.63), meaning it is best at identifying a larger proportion of the actual purchasing sessions in the test set. While the Tuned Random Forest had slightly higher Precision and ROC AUC, the priority of maximizing sales capture through higher Recall led us to select the Tuned LightGBM model.

The ROC AUC score of 0.913 for the selected LightGBM model indicates strong overall discriminative power in separating purchasing from non-purchasing sessions.

## Deployment

The final, tuned LightGBM model has been saved as a robust pipeline artifact (`models/purchase_intent_pipeline.pkl`). This pipeline includes all necessary preprocessing steps, allowing it to take raw session features as input and output a purchase probability or prediction directly.

This pipeline is ready to be deployed:

*   **Batch Processing:** Can be used in an offline process to score daily user sessions and generate lists of high-intent users for targeted email or push notification campaigns (example script in `src/deployment/predict.py`).
*   **Real-time API:** Can be deployed as a web service to provide real-time purchase intent predictions for individual sessions, enabling dynamic website personalization or live chat prompts (example structure in `src/deployment/api/`).

## Repository Structure

```
ShopBoys-Customer-Purchase-Propensity-Predictor/
├── README.md               
├── notebook/               
│   └── purchase_intent_analysis.ipynb 
├── data/                   
│   └── online_shoppers_intention.csv # (Optional: Can include if allowed, otherwise describe how to obtain)
├── src/                   
│   ├── data_preprocessing.py 
│   ├── modeling.py        
│   └── deployment/        
│       ├── predict.py      
│       └── api/           
│           ├── app.py
│           └── requirements.txt
├── models/                
│   ├── purchase_intent_pipeline.pkl 
│   └── (optional: other saved models/transformers from experiments)
├── reports/                
│   ├── eda_summary.pdf     
│   ├── model_evaluation_summary.pdf 
│   └── (optional: presentation slides)
├── .gitignore             
└── requirements.txt
```       

## How to Reproduce

1.  Clone this repository.
2.  Install the required Python packages: `pip install -r requirements.txt`
3.  Place the `online_shoppers_intention.csv` dataset file in the `data/` directory, or modify the notebook to load it from a different location.
4.  Open and run the `notebook/purchase_intent_analysis.ipynb` notebook in a Jupyter environment (like Google Colab or JupyterLab) to follow the analysis steps and reproduce the results.

## Requirements

*   Python 3.7+
*   See `requirements.txt` for specific package versions (pandas, numpy, scikit-learn, matplotlib, seaborn, lightgbm, imbalanced-learn).

## Future Work

*   Implement A/B testing to measure the actual business impact of using the model's predictions.
*   Set up ongoing model monitoring to detect performance degradation or data drift.
*   Explore additional features (e.g., user history beyond a single session, demographics if available, product categories visited).
*   Investigate deep learning models for potential further performance gains.
*   Refine behavioral clustering with different algorithms or features.

## License

[[Apache License]](http://www.apache.org/licenses/LICENSE-2.0)

---
