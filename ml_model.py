import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('purchase_order_events.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print("="*70)
print("PREDICTIVE MODEL: PURCHASE ORDER DELAY PREDICTION")
print("="*70)

# ============================================
# FEATURE ENGINEERING
# ============================================
print("\n📊 FEATURE ENGINEERING...")

# Calculate cycle time for each case
cycle_times = df.groupby('Case_ID').agg({
    'Timestamp': ['min', 'max'],
    'Amount': 'first',
    'Department': 'first',
    'Activity': 'count'
})
cycle_times.columns = ['Start', 'End', 'Amount', 'Department', 'Num_Activities']
cycle_times['Duration_Days'] = (cycle_times['End'] - cycle_times['Start']).dt.total_seconds() / (3600 * 24)

# Feature 1: Has Rework
rework_cases = df[df['Activity'] == 'Rework Request']['Case_ID'].unique()
cycle_times['Has_Rework'] = cycle_times.index.isin(rework_cases).astype(int)

# Feature 2: Is Rejected
rejected_cases = df[df['Activity'] == 'Reject']['Case_ID'].unique()
cycle_times['Is_Rejected'] = cycle_times.index.isin(rejected_cases).astype(int)

# Feature 3: Day of Week
cycle_times['Start_DayOfWeek'] = cycle_times['Start'].dt.dayofweek

# Feature 4: Hour of Day
cycle_times['Start_Hour'] = cycle_times['Start'].dt.hour

# Feature 5: Amount Bracket
cycle_times['Amount_High'] = (cycle_times['Amount'] > 10000).astype(int)
cycle_times['Amount_Medium'] = ((cycle_times['Amount'] > 5000) & (cycle_times['Amount'] <= 10000)).astype(int)

# Feature 6: Department One-Hot Encoding
dept_dummies = pd.get_dummies(cycle_times['Department'], prefix='Dept')
cycle_times = pd.concat([cycle_times, dept_dummies], axis=1)

# Feature 7: Needs Finance Approval
finance_cases = df[df['Activity'] == 'Finance Check']['Case_ID'].unique()
cycle_times['Needs_Finance_Approval'] = cycle_times.index.isin(finance_cases).astype(int)

# Feature 8: Needs Final Approval
final_approval_cases = df[df['Activity'] == 'Final Approval']['Case_ID'].unique()
cycle_times['Needs_Final_Approval'] = cycle_times.index.isin(final_approval_cases).astype(int)

# TARGET: Define "Delayed" as cycle time > median (you can adjust threshold)
threshold_days = cycle_times['Duration_Days'].median()
cycle_times['Is_Delayed'] = (cycle_times['Duration_Days'] > threshold_days).astype(int)

print(f"✅ Features created!")
print(f"   Total Features: {len([col for col in cycle_times.columns if col.startswith(('Has_', 'Is_', 'Num_', 'Start_', 'Amount_', 'Dept_', 'Needs_'))])}")
print(f"   Delay Threshold: {threshold_days:.2f} days")
print(f"   Delayed Cases: {cycle_times['Is_Delayed'].sum()} ({cycle_times['Is_Delayed'].mean()*100:.1f}%)")

# ============================================
# PREPARE TRAINING DATA
# ============================================
print("\n📚 PREPARING TRAINING DATA...")

# Select features for model
feature_cols = [col for col in cycle_times.columns if col.startswith(('Has_', 'Is_', 'Num_', 'Start_', 'Amount_', 'Dept_', 'Needs_'))]
feature_cols = [col for col in feature_cols if col not in ['Is_Delayed', 'Is_Rejected']]  # Remove target and rejected flag

X = cycle_times[feature_cols]
y = cycle_times['Is_Delayed']

# Remove rejected cases from training (they have different behavior)
valid_indices = cycle_times['Is_Rejected'] == 0
X = X[valid_indices]
y = y[valid_indices]

print(f"✅ Training data prepared!")
print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")
print(f"   Class Distribution: {y.value_counts().to_dict()}")

# ============================================
# TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\n📊 Train/Test Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# ============================================
# TRAIN MODEL
# ============================================
print("\n🤖 TRAINING RANDOM FOREST MODEL...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ============================================
# EVALUATE MODEL
# ============================================
print("\n📈 MODEL EVALUATION")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 ROC-AUC Score: {roc_auc:.3f}")

# ============================================
# FEATURE IMPORTANCE
# ============================================
print("\n🔍 FEATURE IMPORTANCE (Top 10)")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'])
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(alpha=0.3)

# 3. Feature Importance
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['Importance'], color='teal')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()

# 4. Prediction Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='On-Time (Actual)', color='green')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='Delayed (Actual)', color='red')
axes[1, 1].set_xlabel('Predicted Probability of Delay')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ml_model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✅ Model evaluation saved as 'ml_model_evaluation.png'")
plt.show()

# ============================================
# SAVE MODEL PREDICTIONS
# ============================================
results = pd.DataFrame({
    'Case_ID': X_test.index,
    'Actual_Delayed': y_test,
    'Predicted_Delayed': y_pred,
    'Delay_Probability': y_pred_proba,
    'Correct_Prediction': (y_test == y_pred).astype(int)
})

results.to_csv('model_predictions.csv', index=False)
print("✅ Predictions saved as 'model_predictions.csv'")

print("\n" + "="*70)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*70)