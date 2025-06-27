# ShopBoys Customer Purchase Propensity Predictor

## Executive Summary

This project developed and evaluated machine learning models to predict online shopper purchase intent for ShopBoys. Facing challenges with inefficient marketing spend and missed sales opportunities, the Marketing and Product teams needed a way to identify users most likely to convert.

Our analysis, using historical session data, uncovered key behavioral drivers of purchase, such as high page engagement, significant page value, and low bounce/exit rates. By clustering user sessions, we identified distinct behavioral segments with varying conversion rates, providing actionable insights for targeted strategies.

Focusing on the business objective to **maximize the identification of potential purchasers (prioritizing Recall)**, we trained and tuned LightGBM and Random Forest models. The **Tuned LightGBM model** demonstrated the best performance in capturing potential buyers (Recall: 0.63) while maintaining strong overall discrimination (ROC AUC: 0.913).

The final model is delivered as a deployable pipeline, ready to integrate into SB's systems for applications like:

*   Identifying high-intent users for targeted promotions.
*   Enabling real-time website personalization.
*   Triggering proactive customer support during high-value sessions.

This project provides ShopBoys with a data-driven tool to enhance sales capture and improve the efficiency of customer engagement efforts.

## Business Problem

ShopBoys (SB), like many e-commerce businesses, invests heavily in driving traffic to its website. However, converting that traffic into sales is a significant challenge. The Marketing team, was finding that broad marketing campaigns resulted in low conversion rates and wasted budget. Simultaneously, the Product team was seeking ways to dynamically enhance the user experience for those showing clear signs of purchase intent. Without a clear signal of *when* a user session was hot, opportunities were being missed.

The core problem was the inability to move beyond reacting to purchases *after* they happened, towards **proactively identifying and influencing sessions *likely* to result in a purchase.** This project was initiated to provide that crucial predictive capability.

## Project Goal & Research Question

The overarching goal was to build and validate a machine learning model capable of accurately predicting purchase intent for individual user sessions on the SB website.

The guiding question was:

> How can we leverage patterns in online shopper behavior and technical session attributes to predict purchase intention, thereby enabling SB to maximize sales opportunities?

## Dataset

The analysis was performed on the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), from the UCI Machine Learning Repository mirroring a typical web session data structure for an online shopping company. It comprises 12,330 sessions with features related to user navigation (page views, durations), engagement metrics (`BounceRates`, `ExitRates`, `PageValues`), technical details (`OperatingSystems`, `Browser`, `TrafficType`), and temporal information (`Month`, `Weekend`, `SpecialDay`, `VisitorType`). The target variable is `Revenue` (True/False). A key characteristic of the dataset, reflecting real-world e-commerce data, is the **significant class imbalance** (only a small percentage of sessions result in revenue).

## Methodology

The project followed a structured data science workflow to derive insights and build a predictive solution:

1.  **Data Loading and Initial Exploration:** Loaded the dataset and performed initial checks (shape, types, summary statistics) to understand the data at a high level and confirm the class imbalance.
2.  **Exploratory Data Analysis (EDA):** Delved deeper into the data through descriptive statistics, correlation analysis, and extensive visualizations. This phase aimed to uncover relationships and patterns indicative of purchase intent.
3.  **Feature Importance Analysis:** Quantified the relative influence of different features on the purchase outcome using a Random Forest model.
4.  **Feature Engineering - Behavioral Clustering:** Applied K-Means clustering to group sessions based on their behavioral profiles. The resulting cluster assignments were added as a new feature, designed to capture nuanced interaction patterns.
5.  **Model Selection, Training, and Evaluation:**
    *   Prepared the data, encoding categorical features and including the engineered cluster feature.
    *   Addressed the target class imbalance in the training data using Random Over-sampling.
    *   Selected and tuned LightGBM and Random Forest models using Randomized Search and cross-validation, optimizing for relevant metrics given the imbalance and the business goal.
    *   Trained the final models and performed a detailed evaluation on a held-out test set, focusing on metrics that reflect the ability to identify purchasing sessions.
6.  **Model Deployment Preparation:** Packaged the selected model and its associated preprocessing steps into a single, deployable pipeline artifact.

## Key Insights from Analysis

The analysis provided critical insights into what drives purchases on SB's simulated platform:

*   **Engagement is Paramount:** Sessions with a higher number of page views and longer durations across Administrative, Informational, and Product-Related pages are significantly more likely to convert.
  
    ![image](https://github.com/user-attachments/assets/677f87d9-177a-4627-b198-759ff123ffc7)


*   **PageValue is a Strong Signal:** `PageValues` shows the strongest direct correlation with `Revenue`. Sessions with higher accumulated page value are more likely to convert.
  
    ![image](https://github.com/user-attachments/assets/fd526ce9-642d-4874-a036-461dd19f7fd4)
   
*   **Reducing Friction is Key:** Low `BounceRates` and `ExitRates` are highly associated with purchases, indicating engaged users who are navigating smoothly rather than leaving quickly.
    *   *(See: Above Box Plots and Correlation Heatmap - negative correlations)*
*   **Distinct User Segments Exist:** K-Means clustering revealed behavioral groups. Cluster 1, characterized by very high engagement and low exit rates (visible in the **box plots by Cluster**), had the highest conversion rate (shown in the **Revenue Proportion by Cluster bar plot**).
  
    ![image](https://github.com/user-attachments/assets/92ef501c-a339-43cd-ab86-9c8694a7875d)
    ![image](https://github.com/user-attachments/assets/b7f17efb-09a4-44c7-b5be-1c232b4251d7)

*   **Timing Matters:** Certain months (like Ocotber, November) and weekdays show higher proportions of purchasing sessions.

    ![image](https://github.com/user-attachments/assets/1fed274d-549f-44bc-91cc-40b54a2571ca)

These insights not only support the model's predictions but also provide actionable guidance for the Product team (e.g., improving navigation, identifying high-value page flows) and the Marketing team (e.g., tailoring campaigns based on engagement levels or cluster membership).

## Model Selection & Performance (Aligning with Business Priority)

Given SB's primary objective to **maximize the identification of potential purchasers (high Recall)**, we evaluated our tuned models with this goal in mind. Both LightGBM and Random Forest performed well, but showed different trade-offs between correctly identifying positives (Recall) and avoiding false alarms (Precision).

After comparing the tuned models on the unseen test set, the **Tuned LightGBM** model was selected because it achieved a higher **Recall (0.63)** for the 'Purchase' class compared to the Tuned Random Forest (0.55). This means it is more effective at capturing a larger percentage of the sessions that actually result in a purchase, aligning directly with the business need to identify *more* potential buyers.

While its Precision (0.68) was slightly lower than Tuned Random Forest (0.75), and its ROC AUC (0.913) was marginally lower than Tuned Random Forest (0.919), the superior Recall addresses the core business requirement.

The overall performance metrics for the selected **Tuned LightGBM model** on the test set are:

| Metric            | Value  | Interpretation                                                                 |
| :---------------- | :----- | :----------------------------------------------------------------------------- |
| Accuracy          | 0.90   | Overall correct predictions (useful context, but not the main focus due to imbalance). |
| Precision (True)  | 0.68   | 68% of sessions predicted as 'Purchase' actually resulted in one.                |
| **Recall (True)** | **0.63** | The model identified 63% of all actual purchasing sessions. (Primary Focus)    |
| F1-Score (True)   | 0.65   | Harmonic mean of Precision and Recall for the 'Purchase' class.              |
| ROC-AUC Score     | 0.913  | Strong overall ability to distinguish purchasing from non-purchasing sessions.   |

The ROC Curve below visually demonstrates the trade-off between True Positive Rate (Recall) and False Positive Rate for the selected model:

   ![image](https://github.com/user-attachments/assets/132d53aa-a61e-449e-8271-faa6fa8dc4e9)
    
## Deployment

The final **Tuned LightGBM Pipeline** has been saved as `models/purchase_intent_pipeline.pkl`. This single artifact encapsulates all necessary preprocessing steps and the trained model.

This pipeline is ready to be integrated into ShopBoys' operational systems:

*   **Batch Scoring:** The pipeline can be used in scheduled jobs to score daily session data, generating lists of high-potential leads for email marketing automation or CRM updates (see example script in `src/deployment/predict.py`).
*   **Real-time API:** The pipeline can be loaded into a web service (e.g., a Flask or FastAPI app) to provide low-latency purchase intent predictions for individual sessions as they happen, enabling dynamic website content, personalized recommendations, or proactive chat invitations (see example structure in `src/deployment/api/`).

## Repository Structure

```
ShopBoys-Customer-Purchase-Propensity-Predictor/
├── README.md               
├── notebook/               
│   └── purchase_propensity_prediction.ipynb 
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

1.  Clone this repository: `git clone https://github.com/ogeohia/ShopBoys-Customer-Purchase-Propensity-Predictor.git` 
2.  Navigate to the repository directory: `cd ShopBoys-Customer-Purchase-Propensity-Predictor`
3.  Install the required Python packages: `pip install -r requirements.txt`
4.  Place the `online_shoppers_intention.csv` dataset file in the `data/` directory, or modify the notebook to load it from your preferred location.
5.  Open and run the `notebook/purchase_intent_analysis.ipynb` notebook in a Jupyter environment (like Google Colab, JupyterLab, or VS Code with the Python extension) to execute the full analysis workflow.

## Requirements

*   Python 3.7+
*   See `requirements.txt` for specific package versions (pandas, numpy, scikit-learn, matplotlib, seaborn, lightgbm, imbalanced-learn).

## Future Work

To further enhance this project and maximize its value for SB:

*   **A/B Testing:** Implement A/B tests to quantitatively measure the uplift in conversion rates resulting from using the model's predictions in marketing campaigns or website personalization.
*   **Model Monitoring:** Establish automated monitoring of the model's performance in production and set up alerts for data drift or prediction quality degradation.
*   **Feature Expansion:** Explore incorporating additional data sources, such as customer purchase history, browsing history across sessions, or product popularity trends.
*   **Advanced Modeling:** Investigate other advanced modeling techniques or ensemble methods.
*   **Threshold Optimization:** Further tune the prediction probability threshold based on the real-world costs of false positives and false negatives in specific deployment scenarios.

## License

[[Apache License]](http://www.apache.org/licenses/LICENSE-2.0)

---

