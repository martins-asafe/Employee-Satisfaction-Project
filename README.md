# Employee Satisfaction & Fulfillment Predictor

**Team:** Generalizers (Asafe Brandao & Jason Shan)

**Documentation:** See `Final-Report.pdf` for the complete technical report.

**Tech:** Python, Scikit-Learn, Pandas, NumPy

**Focus:** Classification, Ensemble Learning, Feature Engineering

## Project Overview
This project builds a machine learning pipeline to predict whether an employee is **Satisfied** or **Fulfilled** based on internal survey data.

Because "unhappy" employees are rare but critical to identify for retention, we prioritized **Weighted Recall** and **Balanced Accuracy** over simple accuracy.

## The Approach
1.  **Preprocessing:** Converted survey timestamps into "duration" features to detect rushed answers.
2.  **Cleaning:** Removed ~190 outliers using DBSCAN (epsilon=5.9).
3.  **Selection:** Reduced 114 features down to 35 using Random Forest importance thresholds, filtering out uninformative survey questions.
4.  **Modeling:** Compared **Voting** vs. **Stacking Ensembles**.
    * *Winner:* **Stacking Classifier** (Logistic Regression meta-learner).

## Results
The dataset was heavily imbalanced with low feature correlation, making prediction difficult.
* **Best Model:** Stacking Ensemble.
* **Satisfaction Balanced Accuracy:** 43.88%.
* **Key Win:** We significantly improved the detection of "Discontent" employees (Recall on Class 0) compared to baseline models.
