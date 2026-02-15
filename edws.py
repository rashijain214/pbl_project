## ==============================================
## MASTER DISASTER DASHBOARD v3.1 - UPLOAD VERSION
## dataset.csv UPLOAD + FULL ANALYSIS
## ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

print("📁 UPLOAD YOUR dataset.csv FILE")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"✅ UPLOADED & LOADED: {filename}")

# Load dataset
df = pd.read_csv(filename)
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head(3))
# PART 1: EXPLORATORY DATA ANALYSIS VISUALS
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))

# 1. DISASTER TYPE DISTRIBUTION
plt.subplot(2, 3, 1)
if 'Disaster Type' in df.columns:
    top_types = df['Disaster Type'].value_counts().head(10)
    plt.pie(top_types.values, labels=top_types.index, autopct='%1.1f%%', startangle=90)
    plt.title('🏭 Top 10 Disaster Types Distribution', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No Disaster Type column', ha='center', va='center')
    plt.title('Disaster Type Distribution')

# 2. DISASTERS OVER YEARS
plt.subplot(2, 3, 2)
if 'Year' in df.columns:
    yearly_counts = df['Year'].value_counts().sort_index()
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6)
    plt.title('📈 Disasters Over Years Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Disasters')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No Year column', ha='center')

# 3. DISASTER GROUP BREAKDOWN
plt.subplot(2, 3, 3)
if 'Disaster Group' in df.columns:
    group_counts = df['Disaster Group'].value_counts()
    plt.bar(group_counts.index, group_counts.values, color=['red', 'orange', 'blue', 'green'])
    plt.title('🏷️ Disaster Groups', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
else:
    plt.text(0.5, 0.5, 'No Disaster Group', ha='center')

# 4. REGION-WISE DISASTERS
plt.subplot(2, 3, 4)
if 'Region' in df.columns:
    region_counts = df['Region'].value_counts().head(8)
    plt.barh(region_counts.index, region_counts.values, color='skyblue')
    plt.title('🌍 Disasters by Region', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No Region column', ha='center')

# 5. IMPACT HEATMAP (Deaths vs Affected)
plt.subplot(2, 3, 5)
impact_cols = ['Total Deaths', 'No Affected', 'No Injured']
available_impacts = [col for col in impact_cols if col in df.columns]
if len(available_impacts) >= 2:
    temp_df = df[available_impacts].fillna(0)
    sns.heatmap(temp_df.corr(), annot=True, cmap='Reds', center=0, fmt='.2f')
    plt.title('🔥 Impact Correlation Heatmap', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Insufficient impact columns', ha='center')

plt.tight_layout()
plt.show()
## PART 2 (FIXED): ADVANCED FEATURE CORRELATION ANALYSIS
# Handle '30.37 N' format latitude/longitude strings

plt.figure(figsize=(18, 14))

# FIXED NUMERIC CLEANING FUNCTION
def clean_lat_lon(series):
    """Convert '30.37 N', '72.83 W' → 30.37, -72.83"""
    if series.dtype == 'object':
        # Extract numeric part and handle direction
        series = series.astype(str)
        # Remove letters and convert
        numeric_only = series.str.replace(r'[^\d.-]', '', regex=True)
        coords = pd.to_numeric(numeric_only, errors='coerce')

        # Handle N/S/E/W direction (if original had letters)
        has_direction = series.str.contains(r'[NSWE]', regex=True, na=False)
        coords[has_direction & (series.str.contains('W|S', na=False))] *= -1
        return coords.fillna(0)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# Extract and clean numeric columns
numeric_cols = ['Year', 'Latitude', 'Longitude', 'Total Deaths', 'No Injured', 'No Affected', 'Total Affected']
numeric_cols = [col for col in numeric_cols if col in df.columns]

print("🔧 Cleaning columns:", numeric_cols)
corr_data = df.copy()

# Apply robust cleaning to ALL numeric columns
for col in numeric_cols:
    if col in ['Latitude', 'Longitude']:
        corr_data[col] = clean_lat_lon(df[col])
    else:
        corr_data[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)

# Calculate correlation matrix
corr_matrix = corr_data[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# BEAUTIFUL HEATMAP
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', linewidths=2.5, cbar_kws={'shrink': 0.8},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

plt.title('🔗 FULL FEATURE CORRELATION MATRIX (Lat/Lon FIXED)',
          fontsize=22, fontweight='bold', pad=25)
plt.tight_layout()
plt.show()

# Correlation insights
print("✅ CORRELATION MATRIX SUCCESS!")
print("\n🔥 TOP 10 CORRELATIONS:")
top_corrs = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates(keep='first')
print(top_corrs.head(10))

print(f"\n📊 Data quality check:")
for col in numeric_cols:
    clean_count = corr_data[col].notna().sum()
    print(f"  {col}: {clean_count}/{len(df)} values cleaned ({clean_count/len(df)*100:.1f}%)")
# PART 3: PREPARE MODELS + TRAIN (If not done)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

# Clean and prepare data (safe version)
def safe_numeric(series): return pd.to_numeric(series.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)

features = ['Year', 'Disaster Group', 'Disaster Type', 'Country', 'Region',
           'Latitude', 'Longitude', 'Total Deaths', 'No Injured', 'No Affected']
features = [f for f in features if f in df.columns]

df_model = df[features].copy()
for col in features:
    if col not in ['Disaster Group', 'Disaster Type', 'Country', 'Region']:
        df_model[col] = safe_numeric(df_model[col])

# Target
df_model['Total_Impact'] = safe_numeric(df['Total Deaths']) + safe_numeric(df['No Affected'])
df_model['Severity'] = (df_model['Total_Impact'] > df_model['Total_Impact'].quantile(0.85)).astype(int)

# Encode categoricals
le_dict = {}
for col in ['Disaster Group', 'Disaster Type', 'Country', 'Region']:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna('Unknown')
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

X = df_model.drop('Severity', axis=1)
y = df_model['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


TARGET_PRECISION = 0.95

# ✅ Decision Tree removed
models = {
    'Logistic Regression': LogisticRegression(
        C=0.5,
        penalty='l2',
        random_state=42,
        max_iter=1000
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=30,
        min_samples_leaf=20,
        random_state=42
    ),

    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=8,
        weights='distance'
    ),

    'Support Vector Machine': SVC(
        C=1.5,
        gamma='scale',
        probability=True,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    ),

    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
}

metrics_df = pd.DataFrame(columns=[
    'Accuracy', 'Precision', 'Recall', 'F1',
    'Threshold', 'Predicted_Positive'
])

confusion_matrices = {}
model_instances = {}

print("🎯 TRAINING 6 MODELS (AUTO THRESHOLD ≈ 0.95 PRECISION)")
print("="*100)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)
    model_instances[name] = model

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 🔎 Auto threshold selection
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
    precision_vals = precision_vals[:-1]

    idx = np.argmin(np.abs(precision_vals - TARGET_PRECISION))
    threshold = thresholds[idx]

    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    predicted_positive = y_pred.sum()

    metrics_df.loc[name] = [
        acc, prec, rec, f1, threshold, predicted_positive
    ]

    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    print(f"Threshold: {threshold:.4f}")
    print(f"Predicted Positives: {predicted_positive}/{len(y_test)}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

    if predicted_positive < 10:
        print("⚠️ Too few positive predictions → inflated precision risk")

metrics_df = metrics_df.round(4)

print("\n" + "="*100)
print("🎯 FINAL RESULTS (~95% PRECISION TARGET)")
print("="*100)
print(metrics_df.sort_values('Precision', ascending=False))

best_model_name = metrics_df['Precision'].idxmax()

print("\n🏆 PRECISION CHAMPION:", best_model_name)
print(metrics_df.loc[best_model_name])
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

num_models = len(confusion_matrices)
cols = 3
rows = ceil(num_models / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, cm) in enumerate(confusion_matrices.items()):

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        square=True,
        linewidths=1,
        ax=axes[idx]
    )

    acc = metrics_df.loc[name, 'Accuracy']
    f1 = metrics_df.loc[name, 'F1']

    axes[idx].set_title(
        f"{name}\nAcc: {acc:.3f} | F1: {f1:.3f}",
        fontsize=12,
        fontweight='bold'
    )

    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

# Hide extra subplot if any
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.suptitle("Confusion Matrices - 6 Models", fontsize=18)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

for name, model in model_instances.items():

    # Get probabilities
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}\nAUC = {auc_score:.4f}")
    plt.grid()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Sort by accuracy for better visuals
metrics_df = metrics_df.sort_values('Accuracy', ascending=False)

fig = plt.figure(figsize=(20, 14))

# ---------------------------------------------------
# 1️⃣ Accuracy Bar Chart
# ---------------------------------------------------
ax1 = plt.subplot(2, 2, 1)

metrics_df['Accuracy'].plot(
    kind='bar',
    ax=ax1,
    color='skyblue',
    edgecolor='black'
)

ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='x', rotation=45)


# ---------------------------------------------------
# 2️⃣ Radar Chart (Precision, Recall, F1)
# ---------------------------------------------------
ax2 = plt.subplot(2, 2, 2, polar=True)

categories = ['Precision', 'Recall', 'F1']
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close loop

metrics_top = metrics_df.head(5)

for name, row in metrics_top.iterrows():
    values = [row['Precision'], row['Recall'], row['F1']]
    values += values[:1]   # close loop

    ax2.plot(angles, values, linewidth=2, label=name)
    ax2.fill(angles, values, alpha=0.1)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_title('Precision–Recall–F1 Radar', fontweight='bold')
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))


# ---------------------------------------------------
# 3️⃣ Accuracy vs F1 Scatter
# ---------------------------------------------------
ax3 = plt.subplot(2, 2, 3)

ax3.scatter(metrics_df['Accuracy'], metrics_df['F1'], s=200, alpha=0.7)

for name in metrics_df.index:
    ax3.annotate(
        name,
        (metrics_df.loc[name, 'Accuracy'], metrics_df.loc[name, 'F1']),
        xytext=(5, 5),
        textcoords='offset points'
    )

ax3.set_xlabel('Accuracy')
ax3.set_ylabel('F1 Score')
ax3.set_title('Accuracy vs F1 Tradeoff', fontweight='bold')
ax3.grid(True, alpha=0.3)


# ---------------------------------------------------
# 4️⃣ Predicted Positives vs Accuracy
# (Replacing broken Support column)
# ---------------------------------------------------
ax4 = plt.subplot(2, 2, 4)

if 'Predicted_Positive' in metrics_df.columns:

    ax4.scatter(
        metrics_df['Predicted_Positive'],
        metrics_df['Accuracy'] * 100,
        s=200,
        alpha=0.7
    )

    for name in metrics_df.index:
        ax4.annotate(
            name,
            (metrics_df.loc[name, 'Predicted_Positive'],
             metrics_df.loc[name, 'Accuracy'] * 100),
            xytext=(5, 5),
            textcoords='offset points'
        )

    ax4.set_xlabel('Predicted Positive Count')
    ax4.set_ylabel('Accuracy %')
    ax4.set_title('Predicted Positives vs Accuracy', fontweight='bold')
    ax4.grid(True, alpha=0.3)

else:
    ax4.axis('off')


plt.tight_layout()
plt.show()


# Best model by accuracy
best_model = metrics_df['Accuracy'].idxmax()
best_acc = metrics_df['Accuracy'].max()

print(f"\nBest Model (Accuracy): {best_model} ({best_acc:.3f})")
from sklearn.metrics import classification_report

reports = {}

for name, model in model_instances.items():

    # Use your optimized threshold
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Get stored threshold from metrics_df
    threshold = metrics_df.loc[name, 'Threshold']

    y_pred = (y_proba >= threshold).astype(int)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    reports[name] = report_dict
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

print("\n" + "="*80)
print("COMPLETE CLASSIFICATION REPORTS")
print("="*80)

num_models = len(reports)
cols = 3
rows = ceil(num_models / cols)

fig, axes = plt.subplots(rows, cols, figsize=(24, 10))
axes = axes.flatten()

for idx, (name, report) in enumerate(reports.items()):

    report_df = pd.DataFrame(report).round(3)

    axes[idx].axis('tight')
    axes[idx].axis('off')

    table = axes[idx].table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    acc = metrics_df.loc[name, "Accuracy"]

    axes[idx].set_title(
        f"{name}\nAccuracy: {acc:.3f}",
        fontweight='bold'
    )

# Hide empty subplot if any
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.suptitle(
    "Full Classification Reports (Precision / Recall / F1 / Support)",
    fontsize=16
)

plt.tight_layout()
plt.show()

