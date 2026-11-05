# CodeNova Task 4 – Customer Segmentation Study

## 1. Project objective
The objective is to group customers with similar purchase-profile and shopping-behaviour characteristics. The project uses data cleaning, feature engineering, scaling, one-hot encoding, K-Means clustering, Elbow analysis, Silhouette evaluation, and PCA visualization.

## 2. Dataset information
- **Dataset title:** Customer Purchase Behavior
- **File:** `data/shopping_trends.csv`
- **Contributor:** Hariharan S.
- **Published:** 15 January 2025 (Version 1)
- **DOI:** 10.17632/xv99b2mtfk.1
- **License:** CC BY 4.0
- **Records:** 3,900
- **Columns:** 19
- **Unique customer IDs:** 3,900
- **Source:** Mendeley Data, Customer Purchase Behavior

The publication date is not evidence that the transactions themselves occurred in 2025. The dataset does not provide a per-record transaction date.

Please retain the dataset attribution and CC BY 4.0 license when redistributing the dataset.

## 3. Project folder structure
```text
CodeNova_Task4_Customer_Segmentation/
│
├── README.md
│
├── data/
│   └── shopping_trends.csv
│
├── notebooks/
│   └── customer_segmentation_study.ipynb
│
├── src/
│   └── customer_segmentation.py
│
├── figures/
│   ├── elbow_method.png
│   ├── silhouette_scores.png
│   ├── pca_customer_segments.png
│   └── customer_segment_sizes.png
│
└── reports/
    ├── customer_segments.csv
    ├── cluster_persona_summary.csv
    ├── cluster_selection_metrics.csv
    └── data_quality_and_model_summary.json
```

## 4. Data preparation and feature selection
1. Load the CSV file and inspect its dimensions, column types, summary statistics, and missing values.
2. Remove exact duplicate rows and trim leading/trailing whitespace from text fields.
3. Convert the selected numeric columns to numeric data types. Missing numeric values, if present, are filled using the column median. The supplied dataset has no missing values or duplicate rows.
4. Exclude `Customer ID` because it is an identifier and does not represent customer behaviour.
5. Use these numeric features:
   - Age
   - Purchase Amount (USD)
   - Review Rating
   - Previous Purchases
6. Use these categorical features:
   - Gender
   - Category
   - Season
   - Subscription Status
   - Discount Applied
   - Promo Code Used
   - Payment Method
   - Shipping Type
   - Frequency of Purchases
7. Scale numeric features with `StandardScaler` and encode categorical features with one-hot encoding before calculating Euclidean distances.

Location, item name, and colour are excluded from the clustering features to avoid adding high-cardinality or less directly comparable descriptive fields.

## 5. Mathematical justification

### 5.1 Standardization
K-Means is distance-based. A feature with a larger numeric scale can dominate the distance calculation. Each numeric feature is standardized using:

\[
z = \frac{x-\mu}{\sigma}
\]

where:
- \(x\) is the original feature value,
- \(\mu\) is the feature mean,
- \(\sigma\) is the feature standard deviation,
- \(z\) is the standardized value.

Categorical features are converted to binary indicator columns using one-hot encoding. The resulting numeric matrix is used by K-Means.

### 5.2 Euclidean distance
K-Means assigns each observation to the nearest cluster centroid. The Euclidean distance between observation \(x_i\) and centroid \(\mu_k\) is:

\[
d(x_i,\mu_k)=\sqrt{\sum_{j=1}^{p}(x_{ij}-\mu_{kj})^2}
\]

where \(p\) is the number of processed features.

### 5.3 K-Means objective
K-Means aims to minimize the within-cluster sum of squared distances:

\[
J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|^2
\]

where:
- \(K\) is the number of clusters,
- \(C_k\) is the set of observations assigned to cluster \(k\),
- \(\mu_k\) is the centroid of cluster \(k\).

The algorithm iteratively assigns observations to the nearest centroid and updates centroids until assignments stabilize or the stopping condition is met.

### 5.4 Elbow method
For each tested value of \(K\), the model's inertia (the objective \(J\) above) is recorded. Inertia normally decreases as \(K\) increases. The Elbow method looks for a point where the reduction in inertia begins to slow, balancing compactness against model complexity.

### 5.5 Silhouette coefficient
For an observation \(i\):
- \(a(i)\) is the average distance from \(i\) to other observations in its own cluster.
- \(b(i)\) is the lowest average distance from \(i\) to observations in another cluster.

The silhouette value is:

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

The score ranges from -1 to +1. Values near +1 indicate better separation, values around 0 indicate overlapping clusters, and negative values can indicate that an observation may be assigned to the wrong cluster. The project calculates silhouette scores on a reproducible sample of up to 1,500 rows to reduce computation time.

### 5.6 PCA visualization
Principal Component Analysis (PCA) projects the processed feature matrix into two dimensions. The first two principal components capture the largest possible share of variance among all two-dimensional linear projections. PCA is used for visualization; the final K-Means model is fitted on the full processed feature matrix, not only on the two PCA components.

## 6. Cluster-count selection and actual results
The notebook evaluates \(K=2,3,\ldots,8\).

| K | Inertia | Sampled silhouette score |
|---:|---:|---:|
| 2 | 32851.08 | 0.133830 |
| 3 | 31139.19 | 0.091747 |
| 4 | 29856.68 | 0.085466 |
| 5 | 28792.32 | 0.078579 |
| 6 | 27955.66 | 0.077842 |
| 7 | 27159.06 | 0.076568 |
| 8 | 26492.13 | 0.075952 |

**Selected cluster count: K = 2.** Among the tested values, \(K=2\) has the highest sampled silhouette score (0.133830). This is weak separation, not a high-quality or strongly separated clustering result. The Elbow plot and cluster profiles should be considered alongside the silhouette score; the score does not prove that two natural customer groups exist.

- Cluster 0: 1,677 customers
- Cluster 1: 2,223 customers
- Variance explained by the first two PCA components: approximately 25.14%

## 7. Customer personas and interpretation
The actual cluster profiles show that the strongest distinction is whether the customer record indicates discount and promo-code use:

| Cluster | Customers | Discount-use rate | Promo-code-use rate | Descriptive persona |
|---:|---:|---:|---:|---|
| 0 | 1,677 | 100% | 100% | Discount and promo-code users |
| 1 | 2,223 | 0% | 0% | Customers with no recorded discount or promo-code use |

The average purchase amount, age, review rating, and previous-purchase count are similar across the two clusters. Therefore, the personas are intentionally narrow and do not claim that one group is higher-value, more loyal, or less repeat-oriented.

### Possible business actions
- For discount and promo-code users, test targeted promotions and measure whether offers increase purchase value or repeat behaviour compared with a control group.
- For customers with no recorded discount or promo-code use, test non-discount engagement such as product recommendations, loyalty benefits, or new-arrival messaging.
- Compare conversion and revenue outcomes before scaling either strategy. These are proposed experiments, not proven causal effects.

## 8. How to run the project

### Google Colab
1. Upload the project folder to Google Drive, or upload the notebook and CSV separately.
2. Open `notebooks/customer_segmentation_study.ipynb` in Google Colab.
3. Keep the expected relative path `../data/shopping_trends.csv`, or update the loading cell if the CSV is uploaded directly into the Colab session.
4. Select **Runtime → Run all**.

### Local Python
Install the dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Run the pipeline from the project root:

```bash
python src/customer_segmentation.py
```

## 9. Limitations
- The dataset contains one customer-level row per customer, not a transaction-level ledger with multiple dated purchases.
- No transaction date is supplied. True RFM **Recency** cannot be calculated. `Previous Purchases` and `Purchase Amount (USD)` are available customer-level fields, not transaction-history-derived Frequency and Monetary metrics.
- The dataset's 2025 publication date does not establish that the records were collected in 2025 or 2026. Do not describe it as live or current transaction data.
- The silhouette score is low (0.133830), indicating substantial overlap or weak separation in the selected feature space.
- The clusters are largely distinguished by discount and promo-code indicators. These related fields can dominate the grouping; cluster labels should not be interpreted as deep customer-value segments.
- One-hot encoding creates a mixed feature space in which categorical indicators contribute to Euclidean distance. Results depend on feature selection and preprocessing.
- K-Means favours compact, approximately spherical clusters and can be sensitive to feature representation and initialization.
- PCA's two-dimensional view retains only about 25.14% of the variance, so the plot is a simplified projection and may hide structure present in the full feature space.
- Personas are descriptive summaries, not predictive labels or causal conclusions.

## 10. Dataset citation
Hariharan, S. (2025). *Customer Purchase Behavior* (Version 1). Mendeley Data. DOI: 10.17632/xv99b2mtfk.1. Licensed under CC BY 4.0.
## Outputs
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/3b43678b-fe13-4c52-8b6c-dca6c0004ab7" />
<img width="1913" height="1133" alt="image" src="https://github.com/user-attachments/assets/39d54082-0bb2-410f-a71d-d45503201a55" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/e469f249-8fa5-47be-aa67-cdde994909c4" />



