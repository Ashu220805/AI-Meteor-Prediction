# **Meteor ML Analysis**

This project performs machine learning analysis on meteor data using a single Jupyter Notebook.
It includes four core tasks:

## **1. Impact Prediction (Classification)**

Predicts whether a meteor is likely to **fall (Fell)** or be **found** based on:

* year
* latitude
* longitude
* mass

Model: **Gradient Boosting Classifier**

---

## **2. Time-Series Forecasting**

Forecasts yearly meteor trends using LSTM models:

* meteor count
* mean mass
* fell ratio

---

## **3. Clustering (Unsupervised Learning)**

Groups meteors into patterns using:

* KMeans
* DBSCAN
* Agglomerative Clustering

Features used:

* year
* reclat
* reclong
* mass (log-transformed)

---

## **4. Anomaly Detection**

Identifies unusual meteors using:

* Isolation Forest
* Autoencoder reconstruction error

Outputs include anomaly scores and top abnormal meteors.

---

## **Dataset Requirements**

The notebook expects a file named:

```
meteor_data.csv
```

with the following columns:

```
name, id, nametype, recclass, mass, fall, year, reclat, reclong, GeoLocation, date
```

---

## **How to Use**

1. Place `meteor_data.csv` in the same folder as the notebook.
2. Install requirements:

   ```
   pip install pandas numpy scikit-learn tensorflow plotly joblib pyarrow
   ```
3. Open the `.ipynb` file and run the cells in order.

---

## **Outputs**

* Impact probability
* Forecasted future values
* Cluster labels
* Anomaly scores

All results are generated inside the notebook.


