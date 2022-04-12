# Finding Donors for CharityML

## Project 1 for Udacity Introduction to ML with TensorFlow Nanodegree

### Description

In this project Udacity provided a Jupyter notebook with some code in it. I Implemented additional functionality necessary to successfully complete this project.  

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The goal of the project is to construct a model that accurately predicts whether an individual makes more than $50,000. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publicly available features.

In this project I picked 3 supervised algorithms (K-Nearest Neighbors, Random Forest, and Gradient Boosting) and compared results. Gradient Boosting algorithm showed the best score.  

On the next step, I tuned this model using GridSearchCV. That helped me to slightly improve the results:

#### Results:

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: | 
| Accuracy Score | 0.8630            |  0.8711         |
| F-score        | 0.7395            |  0.7529         |


Then I analyzed feature importance. Training on subset of data with only 5 most important features slightly decreased accuracy (0.8583) and F-score (0.7223), but significantly reduced the training time.


### Project files

    - README.md                   # project description
    - census.csv (*)              # data set
    - example_submission.csv (*)  # example submission for Kaggle competition
    - finding_donors.ipynb        # project Jupyter notebook
    - kaggle.py                   # functions for preparing submission file for Kaggle competition
    - test_census.csv (*)         # test data for Kaggle competition
    - visuals.py (*)              # supplementary visualization code

(*) - files, provided by Udacity
