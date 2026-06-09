"""
HR Attrition Analysis & Risk Prediction
IBM HR Analytics Dataset — 1,470 employees
Model: Logistic Regression | Accuracy: 65.6% | AUC: 0.676

Author: Adarsh Thupakula | github.com/thupakulaadarsh23
Skills: Python, pandas, scikit-learn, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('IBM_HR_Attrition_Dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Attrition rate: {df['Attrition'].mean()*100:.1f}%")
print("\nDepartment breakdown:")
print(df.groupby('Department')['Attrition'].agg(['count','sum','mean'])
      .rename(columns={'count':'Total','sum':'Left','mean':'Rate'})
      .assign(Rate=lambda x: (x['Rate']*100).round(1)))

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('HR Attrition — Exploratory Analysis', fontsize=14, fontweight='bold')

# Overtime impact
ot_data = df.groupby('OverTime')['Attrition'].mean() * 100
axes[0,0].bar(ot_data.index, ot_data.values, color=['#1D9E75','#E24B4A'])
axes[0,0].set_title('Overtime vs Attrition Rate')
axes[0,0].set_ylabel('Attrition Rate (%)')
for i, v in enumerate(ot_data.values):
    axes[0,0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Job satisfaction
sat_data = df.groupby('JobSatisfaction')['Attrition'].mean() * 100
colors_sat = ['#E24B4A','#EF9F27','#1D9E75','#3B6D11']
axes[0,1].bar(sat_data.index, sat_data.values, color=colors_sat)
axes[0,1].set_title('Job Satisfaction vs Attrition Rate')
axes[0,1].set_xlabel('Satisfaction Level (1=Low, 4=High)')
axes[0,1].set_ylabel('Attrition Rate (%)')

# Income distribution
df.boxplot(column='MonthlyIncome', by='AttritionLabel', ax=axes[0,2])
axes[0,2].set_title('Monthly Income by Attrition')
axes[0,2].set_xlabel('Attrition')
plt.sca(axes[0,2]); plt.title('Monthly Income by Attrition')

# Tenure
tenure_bins = pd.cut(df['YearsAtCompany'], bins=[0,3,7,15,40], labels=['<3yr','3-7yr','7-15yr','15+yr'])
tenure_data = df.groupby(tenure_bins)['Attrition'].mean() * 100
axes[1,0].bar(range(len(tenure_data)), tenure_data.values, color=['#EF9F27','#E24B4A','#1D9E75','#378ADD'])
axes[1,0].set_xticks(range(len(tenure_data)))
axes[1,0].set_xticklabels(tenure_data.index)
axes[1,0].set_title('Tenure vs Attrition Rate')
axes[1,0].set_ylabel('Attrition Rate (%)')

# Department
dept_data = df.groupby('Department')['Attrition'].mean() * 100
axes[1,1].barh(dept_data.index, dept_data.values, color=['#378ADD','#1D9E75','#E24B4A'])
axes[1,1].set_title('Department Attrition Rate')
axes[1,1].set_xlabel('Attrition Rate (%)')

# Age distribution
df[df['Attrition']==1]['Age'].hist(ax=axes[1,2], alpha=0.7, color='#E24B4A', label='Left', bins=20)
df[df['Attrition']==0]['Age'].hist(ax=axes[1,2], alpha=0.5, color='#378ADD', label='Stayed', bins=20)
axes[1,2].set_title('Age Distribution by Attrition')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight')
print("\nEDA charts saved: eda_charts.png")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & MODEL TRAINING
# ─────────────────────────────────────────────
features = [
    'Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
    'WorkLifeBalance', 'EnvironmentSatisfaction', 'DistanceFromHome',
    'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'StockOptionLevel', 'PercentSalaryHike', 'JobLevel',
    'OverTime', 'BusinessTravel', 'MaritalStatus', 'Department'
]

df_model = df[features + ['Attrition']].copy()

# Encode categorical variables
encoders = {}
for col in ['OverTime', 'BusinessTravel', 'MaritalStatus', 'Department']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df_model[features]
y = df_model['Attrition']

# Train/test split — stratified to preserve attrition ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Train attrition rate: {y_train.mean()*100:.1f}%")
print(f"Test attrition rate: {y_test.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Logistic Regression with class_weight='balanced' to handle imbalance
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_sc, y_train)

# ─────────────────────────────────────────────
# 4. MODEL EVALUATION
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Stayed','Left']))

auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC Score: {auc:.3f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
print(f"  True Negatives (correctly predicted stayed): {cm[0,0]}")
print(f"  False Positives (predicted left, actually stayed): {cm[0,1]}")
print(f"  False Negatives (predicted stayed, actually left): {cm[1,0]}")
print(f"  True Positives (correctly predicted left): {cm[1,1]}")

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(coef_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 6. RISK SCORING ALL EMPLOYEES
# ─────────────────────────────────────────────
X_all_sc = scaler.transform(df_model[features])
df['AttritionRiskScore'] = model.predict_proba(X_all_sc)[:, 1]
df['RiskCategory'] = pd.cut(
    df['AttritionRiskScore'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

print("\nRisk Distribution:")
print(df['RiskCategory'].value_counts())

# Save enriched dataset
df.to_csv('IBM_HR_Attrition_WithRisk.csv', index=False)
print("\nSaved: IBM_HR_Attrition_WithRisk.csv")

# ─────────────────────────────────────────────
# 7. ROC CURVE PLOT
# ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#E24B4A', lw=2, label=f'Logistic Regression (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
plt.fill_between(fpr, tpr, alpha=0.1, color='#E24B4A')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — HR Attrition Model', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
print("Saved: roc_curve.png")

# ─────────────────────────────────────────────
# 8. KEY BUSINESS INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("KEY BUSINESS INSIGHTS")
print("="*50)

ot_yes = df[df['OverTime']=='Yes']['Attrition'].mean()*100
ot_no  = df[df['OverTime']=='No']['Attrition'].mean()*100
print(f"\n1. OVERTIME: {ot_yes:.1f}% attrition (OT) vs {ot_no:.1f}% (no OT) — {ot_yes/ot_no:.1f}x multiplier")
print("   → RECOMMENDATION: Cap mandatory overtime, introduce comp-off policy")

low_income = df[df['MonthlyIncome'] < 3000]['Attrition'].mean()*100
print(f"\n2. LOW INCOME: {low_income:.1f}% attrition rate for employees earning < ₹25K/month")
print("   → RECOMMENDATION: Conduct salary band review for bottom quartile")

early_tenure = df[df['YearsAtCompany'] <= 2]['Attrition'].mean()*100
print(f"\n3. EARLY TENURE: {early_tenure:.1f}% attrition in first 2 years")
print("   → RECOMMENDATION: Strengthen onboarding program, assign mentors")

low_sat = df[df['JobSatisfaction'] == 1]['Attrition'].mean()*100
print(f"\n4. SATISFACTION: {low_sat:.1f}% attrition at lowest satisfaction score")
print("   → RECOMMENDATION: Quarterly engagement surveys, skip-level meetings")

print(f"\n5. HIGH-RISK EMPLOYEES: {(df['RiskCategory']=='High').sum()} flagged for immediate review")
print("   → Export IBM_HR_Attrition_WithRisk.csv and filter RiskCategory == 'High'")

print("\n✓ Analysis complete. Files saved:")
print("  - IBM_HR_Attrition_WithRisk.csv (with risk scores)")
print("  - eda_charts.png")
print("  - roc_curve.png")
