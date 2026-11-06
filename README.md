# Drug classification machine learning and pharmaceutical ML classifier using a Python scikit-learn drug model

This repository provides a pharmaceutical ML classifier and Python scikit-learn drug model for drug classification machine learning on chemical element pairs. The system predicts whether two chemical elements can form a viable pharmaceutical compound and generates descriptive treatment summaries with a local language model.

## Architecture

The project consists of three components:

- Feature extraction and model training: `element_drug_predictor.py` loads elemental properties from `elements.json`, generates synthetic element pairs, calculates derived interaction metrics, and trains a scikit-learn pipeline. The pipeline couples a `ColumnTransformer` preprocessor with a `RandomForestClassifier`.
- Text generation: `llm.py` loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` through the Hugging Face `transformers` library to produce plain text summaries of hypothetical medical applications.
- Web interface and API: `app.py` runs a Bottle web server that serves `index.html` and exposes JSON endpoints for pair analysis.

### Data and feature pipeline

The dataset generator creates synthetic pairings from the elements listed in `elements.json`. For each pair, the script computes:

- Numerical differences, sums, products, and ratios for atomic number, electronegativity, group, period, atomic mass, valence electrons, and hypothetical binding affinity.
- Categorical one-hot encodings for element category.
- Binary match indicators for shared group, shared period, and shared category.

The preprocessor applies `StandardScaler` to numerical inputs and `OneHotEncoder` to categorical inputs. A grid search tunes the random forest hyperparameters over 3 cross-validation folds. The trained pipeline saves to `element_drug_interaction_model_full.joblib`.

## Setup

The codebase requires Python 3.10. Install dependencies from `reg.txt`:

```bash
pip install -r reg.txt
```

Download or verify the presence of `elements.json` in the root directory before running training or inference tasks.

## Usage

### Train the model

To generate synthetic data and train the random forest classifier:

```bash
python element_drug_predictor.py train
```

This command produces `synthetic_element_drug_data_full.csv` and serializes the trained model to `element_drug_interaction_model_full.joblib`.

### Command line predictions

Predict interaction viability for a single pair of element symbols:

```bash
python element_drug_predictor.py predict_pair H O
```

Start an interactive shell session to test multiple pairs:

```bash
python element_drug_predictor.py predict
```

Type `quit` to exit the interactive prompt.

### Run the web application

Start the Bottle application server:

```bash
python app.py
```

Open `http://localhost:8080` in a web browser to view the interactive periodic table interface.

The web application exposes these endpoints:

- `GET /`: returns the frontend interface.
- `GET /api/elements`: returns the element dictionary from `elements.json`.
- `POST /api/analyze`: accepts a JSON payload such as `{"element1": "H", "element2": "O"}` and returns classification probabilities along with generated text.

## Configuration

The system uses the following configuration files and parameters:

- `elements.json`: defines elemental properties including electronegativity, valence electrons, and element categories.
- `app.py`: sets the web server host to `localhost` and the port to `8080`.
- `llm.py`: sets the text generation model to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and configures generation parameters including max new tokens set to 512, temperature set to 0.9, and top-p set to 0.9.
- `element_drug_predictor.py`: defines file paths for `synthetic_element_drug_data_full.csv` and `element_drug_interaction_model_full.joblib`.

## Educational disclaimer

This project is an educational model and does not provide medical advice. Predictions and generated texts rely on synthetic data and small language models for study purposes only. Do not use this software for clinical diagnosis, treatment decisions, or pharmaceutical formulation.
