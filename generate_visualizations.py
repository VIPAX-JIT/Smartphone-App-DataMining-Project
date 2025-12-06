import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

# Set style
sns.set(style="whitegrid")

# Define paths
DATA_PATH = "data/Google-Playstore.csv"
OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_data():
    print("Loading and cleaning data...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Rename Rating Count to Reviews
    if 'Rating Count' in df.columns:
        df.rename(columns={'Rating Count': 'Reviews'}, inplace=True)
    
    # Clean Installs
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '').str.replace(',', '')
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # Clean Price
    df['Price'] = df['Price'].astype(str).str.replace('$', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Clean Size
    def clean_size(size):
        size_str = str(size).replace(',', '')
        if 'M' in size_str:
            return float(size_str.replace('M', ''))
        elif 'k' in size_str:
            return float(size_str.replace('k', '')) / 1000
        return np.nan

    df['Size'] = df['Size'].apply(clean_size)
    
    # Impute missing Size with mean of Category
    df['Size'] = df['Size'].fillna(df.groupby('Category')['Size'].transform('mean'))
    
    # Drop remaining critical missing values
    df.dropna(subset=['App Name', 'Category', 'Rating', 'Installs', 'Size', 'Price'], inplace=True)
    
    # Save cleaned data
    df.to_csv(f"{OUTPUT_DIR}/google_cleaned.csv", index=False)
    print(f"Cleaned data saved to {OUTPUT_DIR}/google_cleaned.csv")
    return df

def generate_visualizations(df):
    print("Generating visualizations...")
    
    # 1. Rating Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rating'], kde=True, bins=20)
    plt.title('Distribution of App Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(f"{OUTPUT_DIR}/rating_distribution.png")
    plt.close()
    print("Generated rating_distribution.png")
    
    # 2. Category Counts (Recommended)
    plt.figure(figsize=(12, 8))
    category_counts = df['Category'].value_counts().head(10)
    sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
    plt.title('Top 10 App Categories')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.savefig(f"{OUTPUT_DIR}/category_counts.png")
    plt.close()
    print("Generated category_counts.png")
    
    # 3. Price vs Rating (Recommended)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Price', y='Rating', alpha=0.5)
    plt.title('Price vs Rating')
    plt.xlabel('Price')
    plt.ylabel('Rating')
    plt.savefig(f"{OUTPUT_DIR}/price_vs_rating.png")
    plt.close()
    print("Generated price_vs_rating.png")

def perform_clustering(df):
    print("Performing clustering...")
    
    # Features for clustering
    features = ['Rating', 'Reviews', 'Size', 'Price']
    
    # Handle Reviews (convert to numeric first)
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df.dropna(subset=['Reviews'], inplace=True)
    
    X = df[features].copy()
    
    # Log transform skewed features
    X['Reviews'] = np.log1p(X['Reviews'])
    X['Size'] = np.log1p(X['Size'])
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Rating', y='Reviews', hue='Cluster', palette='viridis', alpha=0.6)
    plt.title('Clustering of Apps (Rating vs Log(Reviews))')
    plt.yscale('log')
    plt.ylabel('Reviews (Log Scale)')
    plt.savefig(f"{OUTPUT_DIR}/cluster_scatter.png")
    plt.close()
    print("Generated cluster_scatter.png")
    
    return df

def perform_prediction(df):
    print("Performing prediction...")
    
    # Target: High Rated (Rating > 4.0)
    df['High_Rated'] = (df['Rating'] > 4.0).astype(int)
    
    # Features
    feature_cols = ['Category', 'Reviews', 'Size', 'Installs', 'Price', 'Content Rating']
    X = df[feature_cols].copy()
    y = df['High_Rated']
    
    # Encode categorical features
    le = LabelEncoder()
    X['Category'] = le.fit_transform(X['Category'])
    X['Content Rating'] = le.fit_transform(X['Content Rating'])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Evaluation
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()
    print("Generated confusion_matrix.png")
    
    # Feature Importance (Recommended)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_cols[i] for i in indices], palette='magma')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
    plt.close()
    print("Generated feature_importance.png")
    
    # Metrics CSV
    metrics_dict = {
        'Model': ['Random Forest'],
        'Accuracy': [accuracy_score(y_test, y_pred)],
        'Precision': [precision_score(y_test, y_pred)],
        'Recall': [recall_score(y_test, y_pred)],
        'F1_Score': [f1_score(y_test, y_pred)]
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(f"{OUTPUT_DIR}/model_performance.csv", index=False)
    print(f"Metrics saved to {OUTPUT_DIR}/model_performance.csv")

def perform_elbow_method(df):
    print("Performing Elbow Method...")
    features = ['Rating', 'Reviews', 'Size', 'Price']
    
    # Handle Reviews (convert to numeric first)
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df.dropna(subset=['Reviews'], inplace=True)
    
    X = df[features].copy()
    
    # Log transform skewed features
    X['Reviews'] = np.log1p(X['Reviews'])
    X['Size'] = np.log1p(X['Size'])
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(f"{OUTPUT_DIR}/elbow_method.png")
    plt.close()
    print("Generated elbow_method.png")

def visualize_association_rules(df):
    print("Visualizing Association Rules...")
    from mlxtend.frequent_patterns import apriori, association_rules
    
    # Prepare data for Apriori
    df_rules = df[['Category', 'Rating', 'Size', 'Price', 'Content Rating']].dropna().copy()
    
    # Binning
    df_rules['Rating_Bin'] = pd.cut(df_rules['Rating'], bins=[0, 3.5, 4.5, 5], labels=['Low', 'Avg', 'High'])
    df_rules['Size_Bin'] = pd.qcut(df_rules['Size'], q=3, labels=['Small', 'Medium', 'Large'])
    df_rules['Price_Bin'] = df_rules['Price'].apply(lambda x: 'Free' if x == 0 else 'Paid')
    
    df_rules.drop(['Rating', 'Size', 'Price'], axis=1, inplace=True)
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_rules)
    
    # Apriori
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    
    if not rules.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=rules['support'], y=rules['confidence'], size=rules['lift'], hue=rules['lift'], palette='viridis', sizes=(20, 200))
        plt.title('Association Rules (Support vs Confidence)')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/association_rules_scatter.png")
        plt.close()
        print("Generated association_rules_scatter.png")
    else:
        print("No association rules found to visualize.")

if __name__ == "__main__":
    df = load_and_clean_data()
    generate_visualizations(df)
    perform_elbow_method(df)
    df = perform_clustering(df)
    perform_prediction(df)
    visualize_association_rules(df)
    print("All tasks completed successfully.")
