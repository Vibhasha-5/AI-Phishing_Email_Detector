# AI Powered Phishing Email Detector

This project uses Machine Learning to detect phishing emails based on their textual content.

***Built By Vibhasha Nagvekar <3***

## Dataset
Due to size and licensing constraints, the dataset is not included in the repository.
It can be downloaded from Kaggle:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset


## Tech Stack
- Python
- Scikit-Learn
- TF-IDF
- Logistic Regression
- Streamlit

## Project Structure
data/ → dataset
models/ → trained model
src/ → source code


## How to Run

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Train the Model
python src/train_phishing.py

### 3. Classify via CLI
python src/classify_email.py -t "Urgent! Verify your account now."

### 4. Run Web App
streamlit run src/app_streamlit.py
