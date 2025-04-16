# 🍷 Wine Quality Classification using Decision Tree

This project is a classification task using Decision Tree to predict the quality of red and white Portuguese "Vinho Verde" wines based on physicochemical features.

The dataset consists of:
- `winequality-red.csv`
- `winequality-white.csv`
- `winequality.names` (metadata and citation info)

---

## 📂 Dataset Source

The datasets used in this project are from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) and originally published in:

> P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
> *Modeling wine preferences by data mining from physicochemical properties.*  
> Decision Support Systems, Elsevier, 47(4):547-553, 2009.  
> [DOI:10.1016/j.dss.2009.05.016](http://dx.doi.org/10.1016/j.dss.2009.05.016)

---

## 🔍 Features Used

To simplify the model, only four features were selected:

- `alcohol`
- `sulphates`
- `citric acid`
- `volatile acidity`

---

## 📊 Project Steps

1. **Load & Combine Datasets**  
   Combine red and white wine datasets and add a `type` column for each.

2. **Feature Selection**  
   Select 4 key features based on known correlation to wine quality.

3. **Data Split**  
   Split data into training and testing sets (80:20).

4. **Model Building**  
   Train a `DecisionTreeClassifier` on the selected features.

5. **Evaluation**  
   - Classification Report (precision, recall, f1-score)
   - Accuracy Score
   - Confusion Matrix (with heatmap)

6. **Visualization**  
   - Class distribution before and after splitting
   - Confusion Matrix heatmap
   - Decision Tree visualization

7. **External Testing**  
   Predict wine quality on a sample data outside the test set.

---

## 📈 Requirements

Make sure you have the following Python packages:

```bash
pip install pandas matplotlib seaborn scikit-learn
