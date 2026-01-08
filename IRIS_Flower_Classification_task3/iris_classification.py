# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import os

# ============================================================================
# LOAD DATASET
# ============================================================================

def load_iris_dataset(file_path='iris_flower.csv'):
    """
    Load the Iris dataset from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the iris dataset CSV file
    
    Returns:
    --------
    pandas.DataFrame: Loaded iris dataset
    """
    try:
        # Load the dataset
        iris_df = pd.read_csv(file_path)
        
        print("✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {iris_df.shape}")
        print(f"📋 Columns: {list(iris_df.columns)}")
        
        return iris_df
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        print("Please make sure the 'iris_flower.csv' file is in the current directory.")
        print("\nCreating sample dataset for demonstration...")
        
        # Create sample dataset if file not found
        return create_sample_dataset()
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return create_sample_dataset()

def create_sample_dataset():
    """
    Create a sample iris dataset if CSV file is not available
    """
    from sklearn.datasets import load_iris
    
    # Load iris dataset from sklearn
    iris_data = load_iris()
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = iris_data.target
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("📝 Created sample Iris dataset for demonstration")
    print("ℹ️ Note: Using actual data file is recommended for full accuracy")
    
    return iris_df

# Load the dataset
print("="*60)
print("🌷 IRIS FLOWER CLASSIFICATION PROJECT")
print("="*60)
print("\n📁 Loading Iris dataset...")

iris_df = load_iris_dataset('iris_flower.csv')

# ============================================================================
# EXPLORE DATASET
# ============================================================================

print("\n" + "="*60)
print("📊 DATASET EXPLORATION")
print("="*60)

# Display basic information
print("\n1️⃣ Dataset Overview:")
print("-" * 30)
print(iris_df.head())
print(f"\nDataset Shape: {iris_df.shape}")

print("\n2️⃣ Dataset Information:")
print("-" * 30)
print(iris_df.info())

print("\n3️⃣ Statistical Summary:")
print("-" * 30)
print(iris_df.describe())

print("\n4️⃣ Check for Missing Values:")
print("-" * 30)
missing_values = iris_df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("✅ No missing values found!")
else:
    print(f"⚠️ Found {missing_values.sum()} missing values")
    # Fill missing values with column mean
    iris_df = iris_df.fillna(iris_df.mean())

print("\n5️⃣ Species Distribution:")
print("-" * 30)
species_distribution = iris_df['species'].value_counts()
print(species_distribution)

# ============================================================================
# DATA VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("📈 DATA VISUALIZATION")
print("="*60)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')

# Color scheme
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
species_colors = {
    'setosa': colors[0],
    'versicolor': colors[1],
    'virginica': colors[2]
}

# 1. Species Distribution Pie Chart
species_counts = iris_df['species'].value_counts()
axes[0, 0].pie(species_counts.values, labels=species_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 0].set_title('Species Distribution', fontweight='bold')
axes[0, 0].set_ylabel('')

# 2. Feature Distributions by Species
features = iris_df.columns[:-1]  # All columns except species

for idx, feature in enumerate(features[:4]):
    row = idx // 2 + 1
    col = idx % 2
    
    for species in iris_df['species'].unique():
        species_data = iris_df[iris_df['species'] == species][feature]
        axes[row, col].hist(species_data, alpha=0.5, label=species, 
                           color=species_colors[species], bins=15)
    
    axes[row, col].set_title(f'{feature} Distribution', fontweight='bold')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend(fontsize=9)

# 3. Correlation Heatmap
correlation_matrix = iris_df[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            fmt='.2f', linewidths=0.5, ax=axes[0, 1])
axes[0, 1].set_title('Feature Correlation Heatmap', fontweight='bold')

# 4. Scatter Plot: Petal Length vs Petal Width
for species in iris_df['species'].unique():
    species_data = iris_df[iris_df['species'] == species]
    axes[0, 2].scatter(species_data[features[2]], species_data[features[3]], 
                      label=species, alpha=0.7, s=60, color=species_colors[species])
axes[0, 2].set_title('Petal Length vs Petal Width', fontweight='bold')
axes[0, 2].set_xlabel('Petal Length (cm)')
axes[0, 2].set_ylabel('Petal Width (cm)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional Pairplot for multivariate analysis
print("\n📊 Generating Pairplot for multivariate analysis...")
sns.pairplot(iris_df, hue='species', palette=species_colors, diag_kind='kde')
plt.suptitle('Pairplot of Iris Features by Species', y=1.02, fontweight='bold')
plt.show()

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print("\n" + "="*60)
print("🔧 DATA PREPROCESSING")
print("="*60)

# Encode species labels
label_encoder = LabelEncoder()
iris_df['species_encoded'] = label_encoder.fit_transform(iris_df['species'])

print("✅ Species encoded to numerical values:")
print("-" * 40)
for i, species in enumerate(label_encoder.classes_):
    print(f"  {species}: {i}")

# Prepare features and target
X = iris_df.drop(['species', 'species_encoded'], axis=1)
y = iris_df['species_encoded']

print(f"\n📊 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📈 Training set size: {X_train.shape[0]} samples")
print(f"🧪 Testing set size: {X_test.shape[0]} samples")

# Check class distribution in splits
print("\n📊 Class distribution in splits:")
print("-" * 40)
train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()

for i, species in enumerate(label_encoder.classes_):
    print(f"  {species}:")
    print(f"    Training: {train_dist[i]} samples ({train_dist[i]/len(y_train)*100:.1f}%)")
    print(f"    Testing: {test_dist[i]} samples ({test_dist[i]/len(y_test)*100:.1f}%)")

# ============================================================================
# MODEL BUILDING
# ============================================================================

print("\n" + "="*60)
print("🤖 MODEL BUILDING & TRAINING")
print("="*60)

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
}

# Dictionary to store results
results = {}

print("\nTraining models...")
print("-" * 40)

for name, model in models.items():
    print(f"\n🎯 Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # Store results
    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_test
    }
    
    print(f"  ✅ Training Accuracy: {train_accuracy:.4f}")
    print(f"  ✅ Testing Accuracy: {test_accuracy:.4f}")
    print(f"  ✅ Cross-validation Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("📊 MODEL EVALUATION")
print("="*60)

# Display detailed classification reports
for name, result in results.items():
    print(f"\n{'='*50}")
    print(f"📋 {name} - Detailed Report")
    print('='*50)
    
    # Classification report
    print("\n📄 Classification Report:")
    print("-" * 40)
    y_pred = result['predictions']
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(report)
    
    # Confusion Matrix
    print("📊 Confusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {name}', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*60)
print("🏆 MODEL COMPARISON")
print("="*60)

# Create comparison dataframe
comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Training Accuracy': result['train_accuracy'],
        'Testing Accuracy': result['test_accuracy'],
        'CV Mean Score': result['cv_mean'],
        'CV Std': result['cv_std']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Testing Accuracy', ascending=False)

print("\n📊 Model Performance Comparison:")
print("-" * 60)
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Visualize model comparison
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Bar plot for accuracy comparison
x = np.arange(len(comparison_df))
width = 0.25

bars1 = ax[0].bar(x - width, comparison_df['Training Accuracy'], width, 
                 label='Training Accuracy', color='#4ECDC4')
bars2 = ax[0].bar(x, comparison_df['Testing Accuracy'], width, 
                 label='Testing Accuracy', color='#FF6B6B')
bars3 = ax[0].bar(x + width, comparison_df['CV Mean Score'], width, 
                 label='CV Score', color='#45B7D1')

ax[0].set_xlabel('Model')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Model Performance Comparison', fontweight='bold')
ax[0].set_xticks(x)
ax[0].set_xticklabels(comparison_df['Model'])
ax[0].legend()
ax[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                  f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Select the best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

# Feature Importance for Random Forest
if hasattr(best_model, 'feature_importances_'):
    print(f"\n🔍 Feature Importance Analysis for {best_model_name}:")
    print("-" * 40)
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Plot feature importance
    ax[1].barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#95E1D3')
    ax[1].set_xlabel('Importance')
    ax[1].set_title(f'Feature Importance - {best_model_name}', fontweight='bold')
    ax[1].invert_yaxis()
else:
    ax[1].text(0.5, 0.5, f'Best Model: {best_model_name}\nAccuracy: {comparison_df.iloc[0]["Testing Accuracy"]:.3f}',
              ha='center', va='center', fontsize=12, transform=ax[1].transAxes)
    ax[1].set_title('Best Model Summary', fontweight='bold')
    ax[1].axis('off')

plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📈 Test Accuracy: {comparison_df.iloc[0]['Testing Accuracy']:.4f}")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

print("\n" + "="*60)
print("🔮 MAKING PREDICTIONS")
print("="*60)

def predict_flower(model, sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict flower species based on measurements
    """
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[0]
        confidence = {label_encoder.inverse_transform([i])[0]: round(prob, 3) 
                     for i, prob in enumerate(probabilities)}
    else:
        confidence = {prediction: 1.0}
    
    return prediction, confidence

# Test cases based on typical measurements
test_cases = [
    # Setosa (small petals)
    {"name": "Typical Setosa", "measurements": [5.1, 3.5, 1.4, 0.2]},
    # Versicolor (medium petals)
    {"name": "Typical Versicolor", "measurements": [6.0, 2.7, 4.5, 1.5]},
    # Virginica (large petals)
    {"name": "Typical Virginica", "measurements": [6.8, 3.0, 5.5, 2.1]},
    # Borderline case
    {"name": "Borderline Case", "measurements": [5.8, 2.8, 4.2, 1.3]},
    # Custom test
    {"name": "Custom Test", "measurements": [6.5, 3.0, 5.2, 2.0]}
]

print("\n📋 Test Predictions:")
print("-" * 60)

for i, test_case in enumerate(test_cases, 1):
    sepal_l, sepal_w, petal_l, petal_w = test_case['measurements']
    
    # Make prediction using best model
    prediction, confidence = predict_flower(best_model, sepal_l, sepal_w, petal_l, petal_w)
    
    print(f"\n🌼 Test Case {i}: {test_case['name']}")
    print(f"   Measurements: SL={sepal_l}, SW={sepal_w}, PL={petal_l}, PW={petal_w}")
    print(f"   🔮 Predicted Species: {prediction}")
    
    if len(confidence) > 1:
        print(f"   📊 Confidence Scores:")
        for species, prob in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
            print(f"      {species}: {prob:.1%}")

# Interactive prediction
print("\n" + "="*60)
print("🎮 INTERACTIVE PREDICTION")
print("="*60)

print("\nWant to test your own measurements?")
print("Enter flower measurements (in centimeters):")

try:
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))
    
    prediction, confidence = predict_flower(best_model, sepal_length, sepal_width, petal_length, petal_width)
    
    print(f"\n🌺 Your flower is predicted to be: {prediction}")
    
    if len(confidence) > 1:
        print("\nConfidence levels:")
        for species, prob in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
            print(f"  {species}: {prob:.1%}")
    
except ValueError:
    print("⚠️ Please enter valid numerical measurements")

# ============================================================================
# SAVE THE MODEL
# ============================================================================

print("\n" + "="*60)
print("💾 SAVING THE MODEL")
print("="*60)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the best model
model_filename = f'models/iris_classifier_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model, model_filename)
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print(f"✅ Model saved as: {model_filename}")
print(f"✅ Label encoder saved as: models/label_encoder.pkl")

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'test_accuracy': comparison_df.iloc[0]['Testing Accuracy'],
    'features': list(X.columns),
    'species_mapping': {int(i): species for i, species in enumerate(label_encoder.classes_)}
}

joblib.dump(metadata, 'models/model_metadata.pkl')
print(f"✅ Model metadata saved as: models/model_metadata.pkl")

# ============================================================================
# CREATE DEPLOYMENT FUNCTION
# ============================================================================

def load_saved_model():
    """
    Load the saved model and related files
    """
    try:
        model = joblib.load(model_filename)
        encoder = joblib.load('models/label_encoder.pkl')
        metadata = joblib.load('models/model_metadata.pkl')
        
        print("✅ Model loaded successfully!")
        return model, encoder, metadata
    except:
        print("⚠️ Could not load saved model. Using current best model.")
        return best_model, label_encoder, {}

def predict_new_flower(measurements):
    """
    Predict flower species for new measurements
    
    Parameters:
    -----------
    measurements : list or array-like
        [sepal_length, sepal_width, petal_length, petal_width]
    
    Returns:
    --------
    dict: Prediction results
    """
    model, encoder, metadata = load_saved_model()
    
    # Ensure measurements are in correct format
    if len(measurements) != 4:
        raise ValueError("Please provide exactly 4 measurements: sepal_length, sepal_width, petal_length, petal_width")
    
    # Create input array
    input_data = np.array([measurements])
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[0]
        confidence = {encoder.inverse_transform([i])[0]: float(prob) 
                     for i, prob in enumerate(probabilities)}
    else:
        confidence = {prediction: 1.0}
    
    return {
        'species': prediction,
        'confidence': confidence,
        'measurements': {
            'sepal_length': measurements[0],
            'sepal_width': measurements[1],
            'petal_length': measurements[2],
            'petal_width': measurements[3]
        }
    }

# Test the deployment function
print("\n" + "="*60)
print("🚀 DEPLOYMENT TEST")
print("="*60)

test_measurement = [5.7, 3.0, 4.2, 1.2]
result = predict_new_flower(test_measurement)

print(f"\nTest Measurement: {test_measurement}")
print(f"Predicted Species: {result['species']}")
print(f"Top Confidence: {max(result['confidence'].values()):.1%}")

# ============================================================================
# PROJECT SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📋 PROJECT SUMMARY")
print("="*60)

print(f"""
🌷 IRIS FLOWER CLASSIFICATION PROJECT SUMMARY:

📊 Dataset:
  - Total samples: {len(iris_df)}
  - Features: {len(X.columns)} measurements
  - Species: {len(label_encoder.classes_)} types

🤖 Models Trained:
  1. Logistic Regression
  2. Random Forest Classifier
  3. K-Nearest Neighbors

🏆 Best Model: {best_model_name}
  - Test Accuracy: {comparison_df.iloc[0]['Testing Accuracy']:.2%}
  - Cross-validation: {comparison_df.iloc[0]['CV Mean Score']:.2%} (±{comparison_df.iloc[0]['CV Std']:.2%})

🔍 Key Findings:
  - Petal measurements are most important for classification
  - Setosa flowers are easiest to distinguish
  - Versicolor and Virginica have some overlapping features

💾 Model Saved:
  - Model file: {model_filename}
  - Ready for deployment!

🎯 How to Use:
  1. Load the saved model using joblib
  2. Provide [sepal_length, sepal_width, petal_length, petal_width]
  3. Get instant species prediction with confidence scores
""")

print("✨ Project completed successfully! ✨")