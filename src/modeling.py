# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Train and Evaluate LightGBM Model with Resampling
print("\n--- Evaluating LightGBM Model with Resampling ---")
lgb_model = lgb.LGBMClassifier(random_state=42, force_row_wise=True)
# Fit the model on the resampled training data using selected features
lgb_model.fit(X_train_selected, y_train_resampled)

# Make predictions on the original (unresampled) test set using selected features
y_pred_lgb = lgb_model.predict(X_test_selected)

# Print evaluation metrics for LightGBM
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LightGBM Classification Report:\n", classification_report(y_test, y_pred_lgb))
# Calculate ROC-AUC score
y_prob_lgb = lgb_model.predict_proba(X_test_selected)[:, 1]
print("LightGBM ROC-AUC Score:", roc_auc_score(y_test, y_prob_lgb))


# Train and Evaluate Random Forest Model with Resampling
print("\n--- Evaluating Random Forest Model with Resampling ---")
rf_model = RandomForestClassifier(random_state=42)
# Fit the model on the resampled training data using selected features
rf_model.fit(X_train_selected, y_train_resampled)

# Make predictions on the original (unresampled) test set using selected features
y_pred_rf = rf_model.predict(X_test_selected)

# Print evaluation metrics for Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
# Calculate ROC-AUC score
y_prob_rf = rf_model.predict_proba(X_test_selected)[:, 1]
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test, y_prob_rf))