# Comparative Data Mining Project: Android vs iOS

This project performs a comprehensive data mining analysis on Google Play Store and Apple App Store datasets. It is structured as a series of Jupyter Notebooks, guiding you through the entire process from data cleaning to predictive modeling.

## Project Structure

- **`data/`**: Contains the raw datasets (`Google-Playstore.csv`, `appleAppData.csv`).
- **`notebooks/`**: The core analysis files.
    - `01_Data_Cleaning.ipynb`: Step-by-step data cleaning and preprocessing.
    - `02_EDA.ipynb`: Exploratory Data Analysis and visualizations.
    - `03_Clustering.ipynb`: K-Means clustering and segmentation.
    - `04_Prediction.ipynb`: Predictive models (KNN, Random Forest) for installs and ratings.
    - `05_Association_Rules.ipynb`: Apriori algorithm for rule mining.
    - `06_Conclusion.ipynb`: Final summary and business recommendations.
- **`output/`**: Generated files (cleaned datasets, models, etc.).

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
    ```

2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3.  **Open the Notebooks**:
    Navigate to the `notebooks/` folder and open `01_Data_Cleaning.ipynb` to start. Run the cells sequentially.

## Key Features
- **Comparative Analysis**: Directly compares Android and iOS markets.
- **Educational Approach**: Each notebook contains detailed markdown explanations of the *theory* and *code*.
- **Advanced Techniques**: Includes Clustering, Random Forest Regression, and Association Rule Mining.
