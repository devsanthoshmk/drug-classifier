import os
import sys
import itertools
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

with open("elements.json") as f:
    ELEMENT_PROPERTIES_DATA = json.load(f)

SYNTHETIC_DATA_FILENAME = "synthetic_element_drug_data_full.csv"
MODEL_FILENAME = "element_drug_interaction_model_full.joblib"

TARGET_COLUMN_NAME = "forms_drug"

NUMERICAL_FEATURES_BASE = ['atomic_number', 'electronegativity', 'group', 'period', 'atomic_mass', 'valence_electrons', 'hypothetical_binding_affinity']
CATEGORICAL_FEATURES_BASE = ['element_category']

def engineer_features_for_pair(element_symbol1, element_symbol2, properties_data):
    if element_symbol1 not in properties_data or element_symbol2 not in properties_data:
        return None

    props1 = properties_data[element_symbol1]
    props2 = properties_data[element_symbol2]
    
    features = {}
    
    for prop_name in NUMERICAL_FEATURES_BASE:
        val1 = props1.get(prop_name)
        val2 = props2.get(prop_name)

        features[f'{prop_name}_1'] = val1 if val1 is not None else np.nan
        features[f'{prop_name}_2'] = val2 if val2 is not None else np.nan
        
        if val1 is not None and val2 is not None:
            features[f'diff_{prop_name}'] = abs(val1 - val2)
            features[f'sum_{prop_name}'] = val1 + val2
            features[f'prod_{prop_name}'] = val1 * val2
            if val2 != 0:
                 features[f'ratio_{prop_name}'] = val1 / val2
            else:
                 features[f'ratio_{prop_name}'] = np.nan 
        else:
            features[f'diff_{prop_name}'] = np.nan
            features[f'sum_{prop_name}'] = np.nan
            features[f'prod_{prop_name}'] = np.nan
            features[f'ratio_{prop_name}'] = np.nan

    for prop_name in CATEGORICAL_FEATURES_BASE:
        features[f'{prop_name}_1'] = props1.get(prop_name, 'Unknown')
        features[f'{prop_name}_2'] = props2.get(prop_name, 'Unknown')
        
    features['same_group'] = 1 if props1.get('group') is not None and props2.get('group') is not None and props1.get('group') == props2.get('group') else 0
    features['same_period'] = 1 if props1.get('period') == props2.get('period') else 0
    features['same_category'] = 1 if props1.get('element_category') == props2.get('element_category') else 0
    
    return features

def generate_synthetic_dataset(properties_data, num_samples_approx=10000): # Increased samples
    sys.stdout.write("Generating synthetic dataset with expanded elements...\n")
    element_symbols = list(properties_data.keys())
    all_possible_pairs = list(itertools.combinations_with_replacement(element_symbols, 2))
    
    if len(all_possible_pairs) == 0:
        sys.stderr.write("No element pairs can be generated. Check element properties data.\n")
        return pd.DataFrame()

    num_to_generate = min(num_samples_approx, len(all_possible_pairs)) 
    
    generated_data = []
    
    if num_to_generate > len(all_possible_pairs): # Should not happen with min
        selected_pairs = random.choices(all_possible_pairs, k=num_to_generate)
    else:
        selected_pairs = random.sample(all_possible_pairs, num_to_generate)

    for el1_sym, el2_sym in selected_pairs:
        if random.random() < 0.5: 
            el1_sym, el2_sym = el2_sym, el1_sym
            
        features = engineer_features_for_pair(el1_sym, el2_sym, properties_data)
        if features is None:
            continue

        forms_drug_score = 0
        
        diff_en = features.get('diff_electronegativity', np.nan)
        sum_hba = features.get('sum_hypothetical_binding_affinity', np.nan)
        prod_ve = features.get('prod_valence_electrons', np.nan)
        cat1 = features.get('element_category_1', 'Unknown')
        cat2 = features.get('element_category_2', 'Unknown')
        diff_am = features.get('diff_atomic_mass', np.nan)

        if pd.notna(diff_en) and 0.5 < diff_en < 2.5: forms_drug_score += 25
        if pd.notna(sum_hba) and sum_hba > 0.8: forms_drug_score += 30
        if pd.notna(prod_ve) and 3 <= prod_ve <= 20 : forms_drug_score += 15
        if (cat1 == 'Nonmetal' and cat2 == 'Nonmetal') or \
           (cat1 == 'Halogen' and cat2 == 'Alkali Metal') or \
           (cat1 == 'Nonmetal' and cat2 == 'Metalloid'): forms_drug_score += 20
        if pd.notna(diff_am) and diff_am < 100 : forms_drug_score += 10
        
        forms_drug_score += random.randint(-20, 20) 
        
        features[TARGET_COLUMN_NAME] = 1 if forms_drug_score > 45 else 0
        generated_data.append(features)
        
    dataset_df = pd.DataFrame(generated_data)
    
    if dataset_df.empty:
        sys.stderr.write("Failed to generate any valid data points for the synthetic dataset.\n")
        return dataset_df

    # Drop rows where key engineered features (diff, sum, prod, ratio) are all NaN
    # This can happen if both elements had None for a base numerical property
    key_feature_prefixes = ['diff_', 'sum_', 'prod_', 'ratio_']
    cols_to_check_for_all_nan = [col for col in dataset_df.columns if any(col.startswith(p) for p in key_feature_prefixes)]
    dataset_df = dataset_df.dropna(subset=cols_to_check_for_all_nan, how='all')


    dataset_df.to_csv(SYNTHETIC_DATA_FILENAME, index=False)
    sys.stdout.write(f"Synthetic dataset with {len(dataset_df)} samples generated and saved to {SYNTHETIC_DATA_FILENAME}\n")
    if not dataset_df.empty and TARGET_COLUMN_NAME in dataset_df:
        sys.stdout.write(f"Class distribution:\n{dataset_df[TARGET_COLUMN_NAME].value_counts(normalize=True)}\n")
    else:
        sys.stdout.write("Could not calculate class distribution for empty or incomplete dataset.\n")
    return dataset_df

def get_feature_names_and_types(sample_df_row):
    numerical_features = []
    categorical_features = []
    for col_name, value in sample_df_row.items():
        if col_name == TARGET_COLUMN_NAME:
            continue
        if isinstance(value, (int, float, np.number)) and pd.notna(value):
            numerical_features.append(col_name)
        elif isinstance(value, str):
            categorical_features.append(col_name)
    return numerical_features, categorical_features

def create_preprocessor(numerical_features, categorical_features):
    transformers = []
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
    
    if not transformers:
        sys.stderr.write("Error: No features identified for preprocessing.\n")
        return None
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
    return preprocessor

def train_model_pipeline(X_data, y_data):
    sys.stdout.write("Starting model training pipeline with expanded element set...\n")
    
    if X_data.empty or y_data.empty:
        sys.stderr.write("Error: Input data for training is empty.\n")
        return None

    # Impute NaNs that might remain in numerical features with the mean of the column
    for col in X_data.select_dtypes(include=np.number).columns:
        if X_data[col].isnull().any():
            X_data[col] = X_data[col].fillna(X_data[col].mean())
    # Impute NaNs in categorical features with 'Unknown'
    for col in X_data.select_dtypes(include='object').columns:
        X_data[col] = X_data[col].fillna('Unknown')


    sample_row = X_data.iloc[0].to_dict()
    numerical_cols, categorical_cols = get_feature_names_and_types(sample_row)

    if not numerical_cols and not categorical_cols:
        sys.stderr.write("Error: No numerical or categorical features found in the dataset after imputation.\n")
        return None

    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    if preprocessor is None:
        return None

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [20, 30, None],
        'classifier__min_samples_split': [2, 5, 10], # Added 10
        'classifier__min_samples_leaf': [1, 3, 5]  # Added 5
    }

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=42, stratify=y_data)
    sys.stdout.write(f"Data split: Training {len(X_train)} samples, Testing {len(X_test)} samples.\n")

    sys.stdout.write("Performing GridSearchCV for hyperparameter tuning (this may take some time)...\n")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        sys.stderr.write(f"Error during GridSearchCV fitting: {e}\n")
        raise

    best_pipeline = grid_search.best_estimator_
    sys.stdout.write(f"Best hyperparameters: {grid_search.best_params_}\n")

    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    sys.stdout.write("\n--- Model Performance Evaluation ---\n")
    sys.stdout.write(f"Training Accuracy: {train_accuracy:.4f}\n")
    sys.stdout.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    sys.stdout.write("Classification Report (Test Set):\n")
    sys.stdout.write(classification_report(y_test, y_pred_test, zero_division=0))
    sys.stdout.write("\nConfusion Matrix (Test Set):\n")
    sys.stdout.write(str(confusion_matrix(y_test, y_pred_test)))
    sys.stdout.write("\n\n")

    try:
        joblib.dump(best_pipeline, MODEL_FILENAME)
        sys.stdout.write(f"Trained model pipeline saved to {MODEL_FILENAME}\n")
    except Exception as e:
        sys.stderr.write(f"Error saving model pipeline: {e}\n")
        raise
        
    return best_pipeline

def predict_element_combination_drug(element_symbol1, element_symbol2):
    # sys.stdout.write(f"Predicting for element combination: {element_symbol1} and {element_symbol2}\n")
    
    if not os.path.exists(MODEL_FILENAME):
        return "Error: Model file not found. Please train the model first by running the script with the 'train' argument."

    try:
        loaded_pipeline = joblib.load(MODEL_FILENAME)
    except Exception as e:
        return f"Error loading model pipeline: {e}"

    
    features_dict = engineer_features_for_pair(element_symbol1, element_symbol2, ELEMENT_PROPERTIES_DATA)
    if features_dict is None:
        return {"drug":False,"score":"1"}
        # return "Error: Could not engineer features for the given element pair."
        
    features_df = pd.DataFrame([features_dict])
    
    for col in features_df.select_dtypes(include=np.number).columns:
        if features_df[col].isnull().any():
            features_df[col] = features_df[col].fillna(0) 
    for col in features_df.select_dtypes(include='object').columns:
        features_df[col] = features_df[col].fillna('Unknown')


    try:
        prediction_numeric = loaded_pipeline.predict(features_df)
        probabilities = loaded_pipeline.predict_proba(features_df)
    except Exception as e:
        return f"Error during prediction: {e}. Ensure input features match training."

    if prediction_numeric[0] == 1:
        return {"drug":True,"score":f"{probabilities[0][1]:.2%}"}
        # return f"The combination of {element_symbol1} and {element_symbol2} is PREDICTED TO FORM A DRUG. Confidence: {probabilities[0][1]:.2%}"
    else:
        return {"drug":False,"score":f"{probabilities[0][1]:.2%}"}
        # return f"The combination of {element_symbol1} and {element_symbol2} is PREDICTED NOT TO FORM A DRUG. Confidence: {probabilities[0][0]:.2%}"

def run_training_process():
    sys.stdout.write("--- Element Drug Combination Prediction Model Training (Full Element Set) ---\n")
    dataset = generate_synthetic_dataset(ELEMENT_PROPERTIES_DATA, num_samples_approx=15000) # Increased samples
    if dataset.empty:
        sys.stderr.write("Training aborted due to empty dataset.\n")
        return False
        
    if TARGET_COLUMN_NAME not in dataset.columns or dataset[TARGET_COLUMN_NAME].isnull().all():
        sys.stderr.write(f"Target column '{TARGET_COLUMN_NAME}' not found or all null in dataset. Training aborted.\n")
        return False
        
    X = dataset.drop(TARGET_COLUMN_NAME, axis=1)
    y = dataset[TARGET_COLUMN_NAME]
    
    if X.empty:
        sys.stderr.write("Feature set X is empty after dropping target. Training aborted.\n")
        return False
    if y.nunique() < 2:
        sys.stderr.write(f"Target column '{TARGET_COLUMN_NAME}' has less than 2 unique values ({y.nunique()} found: {y.unique()}). Stratification and training may fail. Check dataset generation rules. Training aborted.\n")
        return False


    try:
        train_model_pipeline(X, y)
        return True
    except Exception as e:
        sys.stderr.write(f"An error occurred during the training pipeline: {e}\n")
        return False

def interactive_prediction_session():
    # sys.stdout.write("\n--- Interactive Element Drug Prediction (Full Element Set) ---\n")
    if not os.path.exists(MODEL_FILENAME):
        sys.stdout.write("Model not found. Please train the model first using the 'train' command.\n")
        return

    sys.stdout.write("Enter element symbols (e.g., C, O, Na, U, Og). Enter 'quit' to exit.\n")
    
    while True:
        try:
            el1_input = input("Enter symbol for first element: ").strip()
            if el1_input.lower() == 'quit': break
            el1 = el1_input.capitalize() if len(el1_input) < 3 else el1_input[0].upper() + el1_input[1:].lower()


            el2_input = input("Enter symbol for second element: ").strip()
            if el2_input.lower() == 'quit': break
            el2 = el2_input.capitalize() if len(el2_input) < 3 else el2_input[0].upper() + el2_input[1:].lower()


            if not el1 or not el2:
                sys.stdout.write("Both element symbols must be provided.\n")
                continue
                
            result = predict_element_combination_drug(el1, el2)
            sys.stdout.write(f"{result}\n\n")
            
        except KeyboardInterrupt:
            sys.stdout.write("\nExiting interactive session.\n")
            break
        except Exception as e:
            sys.stderr.write(f"An error occurred in interactive session: {e}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "train":
            run_training_process()
        elif command == "predict":
            interactive_prediction_session()
        elif command == "predict_pair" and len(sys.argv) == 4:
            el_sym1_arg = sys.argv[2]
            el_sym1 = el_sym1_arg.capitalize() if len(el_sym1_arg) < 3 else el_sym1_arg[0].upper() + el_sym1_arg[1:].lower()
            el_sym2_arg = sys.argv[3]
            el_sym2 = el_sym2_arg.capitalize() if len(el_sym2_arg) < 3 else el_sym2_arg[0].upper() + el_sym2_arg[1:].lower()
            prediction_output = predict_element_combination_drug(el_sym1, el_sym2)
            sys.stdout.write(prediction_output + "\n")
        else:
            sys.stdout.write(f"Unknown command or incorrect arguments: '{' '.join(sys.argv[1:])}'\n")
            sys.stdout.write("Usage:\n")
            sys.stdout.write("  python element_drug_predictor.py train          - To generate data, train, and save the model.\n")
            sys.stdout.write("  python element_drug_predictor.py predict         - To start an interactive prediction session.\n")
            sys.stdout.write("  python element_drug_predictor.py predict_pair E1 E2 - To predict for a specific pair (e.g., C O).\n")
    else:
        sys.stdout.write("No command specified. Defaulting to training.\n")
        training_success = run_training_process()
        if training_success:
            sys.stdout.write("\nTraining complete. You can now run 'python element_drug_predictor.py predict' for interactive predictions.\n")