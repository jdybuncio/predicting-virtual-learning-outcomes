# New possibilities with the movement toward online learning

## Predicting Virtual Learning Outcomes

Due to the rapid shift towards virtual learning brought on by COVID-19, I am interested in looking into the new possibilities given the additional data virtual learning offers to the education space. I take a dataset from the U.K.'s largest undergraduate education provider, The Open University, to create a prediction model which seeks to identify which students have the highest propensity to fail as of Day 1 of any course. I primarily investigate hypotheses related to if specific interactions with the platform are strong predictors of success/failure.

*by JDyBuncio*
*5/15/2020*

## Accessibility to the Code
To gain access to the cleaned dataframe and the helper functions I created to evaluate my models, one can clone this repository and run the following:
```
git clone https://github.com/jdybuncio/predicting-virtual-learning-outcomes.git
cd predicting-virtual-learning-outcomes/src
python data_processing_script.py
```
This will create the dataframe I used in:  ```data/dataset_for_modeling_day_zero.csv``` 
Helper functions can be imported from: ```src/modeling_script.py```


## Table of Contents
- [Introduction](#introduction)
  - [Background](#background)
  - [The Data](#the-data)
  - [Question and Hypothesis](#question-and-hypothesis)
  - [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Selection](#model-selection)
  - [Test Metric: AUC Score](#Test Metric: AUC)
  - [Feature Selection](#Cross Validation & Hyperparameter Tuning)
  - [Hyperparameter Tuning](#Feature Importance)
  - [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#model-selection)



# Introduction

## Background

<p align="center">
  <img src="images/ny_times_headline.png">
</p>

As the headline shows above, several universities across the world are grappling with the impacts of coronavirus to their normal, in-person, curricula. Several universities, such as all those belonging to the Cal-State umbrella, have already announced that all courses offerred in the Fall of 2020 will be remote. This presents several challenges and a large change. But, it also is an opportunity to learn more about student behavior since on-line courses allow for increased data collection. For example, in order to understand how a student is performing, one normally would rely on a students' homework and test scores and a qualitative metrics such as class participation. What is not used though are all the hours spent outside of the classroom which could serve as strong predictors of success or failure. 

A movement to on-line courses would add to one's ability to gauge how a student is performing. Specificially, the use of cookies would allow a university to further track students' interactions with material. While this would represent a "new" normal for students who have not engaged in on-line courses in the past, it also would open up increased possibilities to identify struggling students earlier and intervene.

In that context, the following work seeks to use only information we have as of Day 1 of an on-line class and create a prediction model to try to identify who is at-risk of failing in the hopes of intervening earlier.


## The Data

The Open University is the largest undergraduate provider in the U.K.. They have made available a dataset [link](https://analyse.kmi.open.ac.uk/open_dataset) which covers 7 modules which were presented 2-4 times across 4 terms: Feb 2013, Oct 2013, Feb 2014, and Oct 2014. The dataset contains 7 tables which contain information on the: Course, Assessments, Students, Registration, Scores, VLE material, and VLE Logs. VLE stands for virtual learning environment, and the data offerred consists of what types of on-line resources are made available in each presentation of the course and, most interestingly, students' daily behvaior with each source down to the click level. For example, this dataset has if a student clicked on the homepage 5 times on the 8th day of a module presentation. The dataset covers 32k+ student outcomes across the 7 modules.

## Question and Hypothesis

Can one predict who will Fail the course on the first day of a class?

I hypothesize that using a combination of student demographics, course information, and students' interaction with materials before the class begin will allow me to predict a student's probability of failing on the first day of a given module-presentation.

Potential Application
* Educator perspective 
  * Being able to identify who is at-risk on the first day of class would allow an education to create intervention strategies early.
* Administrator perspective
  * Adminstrators may be concerned with maximizing re tention in hopes of maximizing tuition. They also could be interested in maximizing graduation rates for rankings purposes.


## Methodology

1. Load in and perform EDA into each of the 7 tables provided. This work can be followed by switching to the exploratory_branch of this repository which has a python notebook which contains a summary of my findings.
2. Merge the dataframes together and perform feature engineering on the VLE level data. Perfom additional EDA.
3. Create a baseline model using univariate analysis. For example, using a student's highest education level, to what degree of accuracy could I predict if a student would fail a given module-presentation.
4. Train and test: Logistic Regression, Random Forest, Ada Boost, and Gradient Boosting model via Cross Validation. Evaluate models based off of the highest AUC scores to guide in model tuning and feature selection. This work can be followed by switching to the modeling_branch of this repository which has a python notebook which contains a summary of my findings.


[Back to Top](#Table-of-Contents)

# Exploratory Data Analysis Highlights

* *Leakage Considered*

I can only consider data I have as of Day 1 of a module-presentation starting to prevent any potential leakage. Some students though already withdraw from a class before Day 1 and are marked as failing the course. I removed these students - 3k students, 9% of dataset -  and that is reflected in all subsequent analysis.

<p align="center">
  <img src="images/unregistration_histo.png" width = 400>

* *Class Balance*

53% of the students Passed and 47% of the students Failed the given module-presentations I observed. This represents a fairly balanced class which means I do not need to employ any over/under sampling techniques.

<p align="center">
  <img src="images/class_balance.png" width = 400>

* *Relationships to Pass/Fail Rates across some demographic variables*

Females have slightly higher pass rates than Males in the dataset I had. Those with higher levels of education and who took the class from higher income areas have higher pass rates as well.

<p align="center">
  <img src="images/demographic_fail_rate.png" width = 800>
</p>

* *Relationships to Pass/Fail Rates to VLE interactions*

I created a feature which marks the number of days before Day 1 of a class that a student interacts with the VLE materials. For example, if a student clicked on any one of the available materials of a module-presentation across 40 days before a class, I would mark them as having 40 days with the VLE materials. This graph served as the basis of my central hypothesis since it shows the largest disparity in fail rates that I saw across any single variable. Student who spent more days with the material before the class began has much lower fail rates than those who did not interact with the material at all.

<p align="center">
  <img src="images/days_w_material_and_fail_rate.png" width = 600>
</p>



# Model Selection

## Test Metric: AUC

The metric I chose to evaluate my models was to optimize the AUC (area-under-the-curve) since I want to:
* Maximize TPR: Predict maximum % students who Fail
* Minimize FPR: Minimize % of students predicted to fail, who Pass since doing so minimizes potential intervention costs.

## Cross Validation & Hyperparameter Tuning 
I used Cross Validation to evaluate the AUC of each of my models in order to direct my hyperparameter tuning and feature selection. 

I also used SKLearn's GridSearch to find the best values for the hyperparameters in my Random Forest and Boosting models.

## Chosen Model
My Gradient Boost Model which included all the VLE interaction features I created had the highest AUC (0.77 - navy line below). This proved to be the same when I used my model to predict my validation set and also the testing set. The parameters of this model were:
n_estimators = 400, learning_rate = 0.2
                                      ,min_samples_split = 5
                                      ,min_samples_leaf = 100
                                      ,max_depth = 3
                                      ,max_features = 'sqrt'
                                      ,subsample = 1


<p align="center">
  <img src="images/roc_auc.png" width = 600>
</p>

## Feature Importance
Below shows the Top 10 Features measured using SKLearn's feature importance from my chosen Gradient Boosting model.

<p align="center">
  <img src="images/feature_importance_gb.png" width = 600>
</p>


## Results and Interpretation


# Conclusion

[Back to Top](#Table-of-Contents)


