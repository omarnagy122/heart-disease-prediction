import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer 
from category_encoders import BinaryEncoder
import joblib 

# --- 1. Load Data ---
df = pd.read_csv('HeartDiseaseTrain-Test.csv')
print("Data loaded. Shape:", df.shape)
# print(df.head())

# --- 2. Define Features and Target ---
Y = df['target']
X = df.drop(columns='target')

# --- 3. Split Data ---
x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=24)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=24)
print("Data splits created.")

# --- 4. Define Preprocessing Steps ---
numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
binary_features = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina']
label_features = ['slope', 'vessels_colored_by_flourosopy']
onehot_features = ['chest_pain_type', 'rest_ecg', 'thalassemia']

scaler = StandardScaler()
binary_encoder = BinaryEncoder(cols=binary_features, drop_invariant=True)
ordinal_encoder = OrdinalEncoder()
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')



# --- 5. Create ColumnTransformer (Preprocessing Only) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('bin', binary_encoder, binary_features),
        ('ord', ordinal_encoder, label_features),
        ('cat', onehot_encoder, onehot_features)
    ],
    remainder='drop'  #remove other columns not specified
)
# --- 6. Fit Preprocessor and Transform Data to find low-correlation features ---
x_train_temp_processed = preprocessor.fit_transform(x_train)
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(onehot_features)
all_feature_names = (list(numerical_features) + 
                     [f"{col}_0" for col in binary_features] + 
                     list(label_features) + 
                     list(ohe_feature_names))

#validate feature names count
if len(all_feature_names) != x_train_temp_processed.shape[1]:
    print(f"Warning: Feature name count ({len(all_feature_names)}) doesn't match data shape ({x_train_temp_processed.shape[1]})")
    all_feature_names = [f"feature_{i}" for i in range(x_train_temp_processed.shape[1])]

# make a DataFrame for correlation analysis
df_temp_for_corr = pd.DataFrame(x_train_temp_processed, columns=all_feature_names)
df_temp_for_corr['target'] = y_train.reset_index(drop=True)

# --- 7. Feature Selection based on Correlation ---
threshold_target = 0.05
target_corr = df_temp_for_corr.corr()['target']
low_corr_features = target_corr[abs(target_corr) < threshold_target].index.tolist()
low_corr_features = [f for f in low_corr_features if f != 'target'] 

print("Features with low correlation to the target (to be dropped):")
print(low_corr_features)


# --- 9. Create the Full Preprocessing Pipeline (with Feature Selection) ---
full_preprocessing_pipeline = Pipeline([
    ('initial_preprocessor', preprocessor),
    ('pca', PCA(n_components=6)) 
])

# --- 10. Fit and Transform the Data using the Full Pipeline ---
print("\nFitting and transforming data with full preprocessing pipeline...")
x_train_processed = full_preprocessing_pipeline.fit_transform(x_train)
x_val_processed = full_preprocessing_pipeline.transform(x_val)
x_test_processed = full_preprocessing_pipeline.transform(x_test)

print("\nProcessed data shapes:")
print("x_train_processed:", x_train_processed.shape)
print("x_val_processed:", x_val_processed.shape)
print("x_test_processed:", x_test_processed.shape)

# --- 11. Model Training and Evaluation ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=24),
    "Random Forest": RandomForestClassifier(random_state=24),
    "SVM": SVC(probability=True, random_state=24)
}

print("\n--- Model Evaluation on Validation Set ---")
best_model_name = None
best_auc = 0
best_model = None

for name, model in models.items():
    model.fit(x_train_processed, y_train)
    y_val_pred = model.predict(x_val_processed)
    y_val_prob = model.predict_proba(x_val_processed)[:, 1]

    precision = precision_score(y_val, y_val_pred)
    accuracy = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_prob)

    print(f"{name}:")
    print(f"  Precision: {precision*100:.3f}%")
    print(f"  Accuracy : {accuracy*100:.3f}%")
    print(f"  ROC-AUC  : {auc*100:.3f}%\n")
    
   # choose the best model based on AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = model

print(f"\n--- Best Model on Validation Set: {best_model_name} (AUC: {best_auc*100:.3f}%) ---")

print("\n--- Model Evaluation on Test Set (using best model) ---")

y_test_pred = best_model.predict(x_test_processed)
y_test_prob = best_model.predict_proba(x_test_processed)[:, 1]

precision = precision_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_prob)

print(f"Best Model ({best_model_name}) on Test Set:")
print(f"  Precision: {precision*100:.3f}%")
print(f"  Accuracy : {accuracy*100:.3f}%")
print(f"  ROC-AUC  : {auc*100:.3f}%\n")

# --- 12. Create and Save the Final Pipeline ---

final_pipeline = Pipeline([
    ('full_preprocessor', full_preprocessing_pipeline), 
    ('classifier', best_model) 
])

# Retrain the final pipeline on the entire training + validation set
X_full_train = pd.concat([x_train, x_val], ignore_index=True)
y_full_train = pd.concat([y_train, y_val], ignore_index=True)

print("\nTraining the final pipeline on all training data...")
final_pipeline.fit(X_full_train, y_full_train)
print("Final pipeline trained.")

# --- 13. Save the Final Model Pipeline ---
joblib.dump(final_pipeline, 'heart_disease_model_pipeline.pkl')
print("\nFinal model pipeline saved as 'heart_disease_model_pipeline.pkl'")
