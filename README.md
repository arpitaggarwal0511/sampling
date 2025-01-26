# **Credit Card Fraud Detection Using Multiple ML Algorithms**

## **Overview**

This project is designed to analyze and detect fraudulent credit card transactions with the help of machine learning algorithms. The dataset is heavily skewed, with fraudulent transactions forming only a small portion of the data. To address this issue, different sampling techniques like SMOTE and other methods are applied to balance the dataset. Multiple machine learning models are trained and evaluated to identify the most effective one for each data subset.

## **Table of Contents**

- [Setup Instructions](#setup-instructions)
- [Project Summary](#project-summary)
- [Workflow Steps](#workflow-steps)
  - [Data Import and Exploration](#data-import-and-exploration)
  - [Target Class Analysis](#target-class-analysis)
  - [Checking for Missing Values](#checking-for-missing-values)
  - [Balancing Dataset with SMOTE](#balancing-dataset-with-smote)
  - [Sample Creation](#sample-creation)
  - [Training and Evaluating Models](#training-and-evaluating-models)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)

## **Setup Instructions**

To execute the code, ensure Python 3.x is installed. Install the required Python packages by following these steps:

1. Clone this repository or download the files.
2. Install the necessary libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## **Project Summary**

This project applies machine learning techniques to identify fraudulent credit card transactions. Due to the class imbalance in the dataset (fewer fraudulent cases), balancing techniques like SMOTE (Synthetic Minority Oversampling Technique) are employed. Five machine learning models are trained and tested on the balanced dataset to determine the most effective approach for detecting fraud.

## **Models Used**

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (k-NN)

### Key Techniques:

- SMOTE for addressing class imbalance.
- Sampling techniques to create diverse subsets for model training.

## **Workflow Steps**

### 1. Data Import and Exploration

The dataset is imported using pandas, and initial exploration is conducted to understand its structure:

```bash
data.head()
data.info()
data.describe()
```

- `data.head()`: Displays the first five rows of the dataset.
- `data.info()`: Shows metadata like column names and data types.
- `data.describe()`: Provides summary statistics for numeric columns.

### 2. Target Class Analysis

Analyze the distribution of the target variable (`Class`) to understand the dataset's imbalance:

```bash
data['Class'].value_counts()
```

### 3. Checking for Missing Values

Identify any missing values in the dataset:

```bash
missing_values = data.isnull().sum()
```

### 4. Balancing Dataset with SMOTE

Use SMOTE to generate synthetic samples for the minority class (fraud cases) and balance the dataset:

```bash
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
```

### 5. Sample Creation

Create five data samples using different sampling techniques:

```bash
# Simple Random Sampling
sample1 = balanced_data.sample(frac=0.2, random_state=1)

# Stratified Sampling
groups = balanced_data.groupby('Class')
sample2 = groups.apply(lambda x: x.sample(frac=0.2, random_state=2)).reset_index(drop=True)

# Systematic Sampling
interval = len(balanced_data) // int(0.2 * len(balanced_data))
start_point = np.random.randint(0, interval)
sample3 = balanced_data.iloc[start_point::interval]

# Cluster Sampling
cluster_labels = np.arange(len(balanced_data)) % 5
balanced_data['Cluster'] = cluster_labels
chosen_cluster = np.random.choice(5)
sample4 = balanced_data[balanced_data['Cluster'] == chosen_cluster].drop('Cluster', axis=1)

# Bootstrapping
sample5 = balanced_data.sample(n=int(0.2 * len(balanced_data)), replace=True, random_state=3)
```

### 6. Training and Evaluating Models

Train multiple machine learning models on the created samples and evaluate their performance:

```bash
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier()
}

results = {}
samples = [sample1, sample2, sample3, sample4, sample5]

for model_name, model in models.items():
    results[model_name] = []
    for i, sample in enumerate(samples):
        X_sample = sample.drop('Class', axis=1)
        y_sample = sample['Class']
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name].append(accuracy)

results_df = pd.DataFrame(results, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
print(results_df)
results_df.to_csv("model_accuracies.csv")
```

## **Dependencies**

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn

## **How to Use**

Install the required libraries by running the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Run the script to:

1. Load and explore the dataset.
2. Preprocess the data, including visualizations.
3. Apply SMOTE to balance the dataset.
4. Generate five diverse samples using different techniques.
5. Train five different machine learning models on each sample.
6. Evaluate and determine the best-performing model for each sample.
