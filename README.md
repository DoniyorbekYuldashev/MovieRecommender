Movie Recommender Project
This project implements a movie recommendation system using the MovieLens dataset. It combines supervised learning (XGBoost, RandomForest, KNN, DecisionTree) and reinforcement learning (Q-Learning, UCB) to recommend movies to users based on their ratings and preferences.
Project Overview
The system processes user ratings, movie metadata, and user demographics to predict ratings and optimize recommendations. Key features include:

Data Preprocessing: Convert .dat files to .csv, extract temporal features (year, month, weekday), and clean data.
Supervised Learning: Train regression models to predict user ratings.
Reinforcement Learning: Use Q-Learning and UCB to optimize movie recommendations.
Visualization: Analyze rating distributions and model performance with Matplotlib and Seaborn.

Folder Structure

data/raw/Original .dat files: ratings.dat, movies.dat, users.dat.
data/preprocessed/ .csv files: test_data.csv, train_data.csv.
notebooks/Jupyter Notebook: MovieRecommenderProject.ipynb (main analysis and models).
src/Python scripts:

requirements.txtPython dependencies.
.gitignoreExcludes unnecessary files (e.g., .ipynb_checkpoints/).

Dataset

Source: MovieLens dataset (https://grouplens.org/datasets/movielens/).
Files:
ratings.dat: UserID, MovieID, Rating, Timestamp.
movies.dat: MovieID, Title, Genres.
users.dat: UserID, Gender, Age, Occupation, Zip-code.


Note: The .dat files are included in data/raw/. The notebook converts them to .csv files in data/processed/. The MIT License applies only to the code, not the dataset, which is subject to GroupLens’s terms.

Set Up Locally
Follow these steps to set up the project locally:

Clone the Repository:
git clone https://github.com/DoniyorbekYuldashev/MovieRecommenderProject.git
cd MovieRecommenderProject


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install Dependencies:
pip install -r requirements.txt


Launch Jupyter Notebook:
jupyter notebook notebooks/MovieRecommenderProject.ipynb



Run in Google Colab
To run the project in Google Colab:

Upload the repository files or clone:
!git clone https://github.com/DoniyorbekYuldashev/MovieRecommenderProject.git


Install dependencies:
!pip install -r MovieRecommenderProject/requirements.txt


Upload data/raw/*.dat files to Colab’s file system.

Open notebooks/MovieRecommenderProject.ipynb and run the cells.


Usage

Explore the Notebook: Run MovieRecommenderProject.ipynb to:
Convert .dat files to .csv.
Preprocess data (e.g., extract datetime features).
Train ML models and evaluate performance (MSE, R², MAE).
Apply Q-Learning and UCB for recommendations.
Visualize results (e.g., rating distributions).


Use Scripts: Import functions from src/ for modular tasks:
preprocess.py: Data conversion and cleaning.
models.py: Model training.
recommend.py: RL-based recommendations.


Requirements

Python 3.8+
Libraries (see requirements.txt):
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
xgboost==2.0.3


Example Workflow

Run the notebook’s import cell to load libraries.
Convert .dat files to .csv using preprocess.py functions.
Preprocess data (e.g., extract Year, Month, Weekday).
Train models (XGBoost, RandomForest, etc.) and evaluate metrics.
Run Q-Learning/UCB to generate recommendations.
Visualize results with plots.

Notes

Large Files: If .dat files exceed GitHub’s 100 MB limit, download ml-1m.zip from https://grouplens.org/datasets/movielens/1m/, extract ratings.dat, movies.dat, and users.dat, and place them in data/raw/.
Colab Users: Adjust file paths (e.g., /content/MovieRecommenderProject/data/raw/) and install dependencies to match requirements.txt.
Reproducibility: Use requirements.txt for consistent library versions.

Acknowledgments

MovieLens Dataset: Provided by GroupLens (https://grouplens.org/datasets/movielens/).
Libraries: Thanks to pandas, scikit-learn, xgboost, and others.

Contact
For questions or contributions, open an issue or pull request on GitHub.
