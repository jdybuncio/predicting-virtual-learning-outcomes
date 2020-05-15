# New possibilities with the movement toward online learning

## Predicting Virtual Learning Outcomes

Due to the rapid shift towards virtual learning brought on by COVID-19, I am interested in looking into the new possibilities given the additional data virtual learning offers to the education space. I take a dataset from the U.K.'s largest undergraduate education provider, The Open University, to create a prediction model which seeks to identify which students have the highest propensity to fail as of Day 1 of any course. I primarily investigate hypotheses related to if specific interactions with the platform are strong predictors of success/failure.

*by JDyBuncio*
*5/15/2020*


## Table of Contents
- [Introduction](#introduction)
  - [Background](#background)
  - [The Data](#the-data)
  - [Question and Hypothesis](#question-and-hypothesis)
  - [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Selection](#model-selection)
  - [Test Metric: F1 Score](#test-metric-f1-score)
  - [Feature Selection](#feature-selection)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Chosen Model](#chosen-model)
  - [Specifications](#specifications)
  - [Model Assessment](#model-assessment)
  - [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#model-selection)
- [Citation](#citation])


# Introduction

## Background

<p align="center">
  <img src="images/ny_times_headline.png">
</p>

As the headline shows above, several universities across the world are grappling with the impacts of coronavirus to their normal, in-person, curricula. Several universities, such as all those belonging to the Cal-State umbrella, have already announced that all courses offerred in the Fall of 2020 will be remote. This presents several challenges and a large change. But, it also is an opportunity to learn more about student behavior since on-line courses allow for increased data collection. For example, in order to understand how a student is performing, one normally would rely on a students' homework and test scores and a qualitative metrics such as class participation. What is not used though are all the hours spent outside of the classroom because there is no way to track them in an in-person setting. 

A movement to on-line courses would add to one's ability to gauge how a student is performing. Specificially, the use of cookies would allow a university to further track students' interactions with material. While this brings a lot of privacy concerns with it and would represent a "new" normal for students who have not engaged in on-line courses in the past, it also would open up increased possibilities to identify struggling students earlier and intervene.

In that light, the following work seeks to use only information we have as of Day 1 of an on-line class and create a prediction model to try to identify who is at-risk of failing in the hopes of intervening earlier.


## The Data

The Open University is the largest undergraduate provider in the U.K.. They have made available a dataset [link](https://analyse.kmi.open.ac.uk/open_dataset) which covers 7 modules which were presented 2-4 times across 4 terms: Feb 2013, Oct 2013, Feb 2014, and Oct 2014. The dataset contains 7 tables which contain information on the: Course, Assessments, Students, Registration, Scores, VLE material, and VLE Logs. VLE stands for virtual learning environment, and the data offerred consists of what types of on-line resources are made available in each presentation of the course and, most interestingly, students' daily behvaior with each source down to the click level. For example, this dataset has if a student clicked on the homepage 5 times on the 8th day of a module presentation. The dataset covers 32k+ student outcomes in each module-presentation.

## Question and Hypothesis

Can one predict who will Fail the course on the first day of class?

I hypothesize that using a combination of student demographics, course information, and students' interaction with materials before the class begin will allow me to predict a student's probability of failing on the first day of a given module-presentation.

Potential Application
* Educator perspective 
  * Being able to identify who is at-risk on the first day of class would allow an education to create intervention strategies early.
* Administrator perspective
  * Adminstrators may be concerned with maximizing re tention in hopes of maximizing tuition. They also could be interested in maximizing graduation rates for rankings purposes.


## Methodology

<p align="center">
  <img src="images/methodology.png" width = 800>
</p>


MVP
1. Encode features so that data can be analyzed via logistic regression
2. Leverage Cross Validation to aid in model/feature selection
3. Using the CV selected model, conduct logistic regression analysis to explore how well the 'best' model can predict relationship status using educational outcomes/characteristics


[Back to Top](#Table-of-Contents)

# Exploratory Data Analysis

### Groups

* 

### Feature Categories


<p align="center">
  <img src="images/unregistration_histo.png" width = 400>


<p align="center">
  <img src="images/class_balance.png" width = 400>


<p align="center">
  <img src="images/demographic_fail_rate.png" width = 400>
</p>



<p align="center">
  <img src="images/days_w_material_and_fail_rate.png" width = 400>
</p>

)

# Model Selection




### Cross Validation


## Test Metric: AUC


## Feature Selection
Evaluating the performance of x models with varying features:



## Hyperparameter Tuning 
Used SKLearn's GridSearch to find the best values for the following hyperparameters.




# Chosen Model

## Specifications
 

## Model Assessment

### CV & Performance Metrics



### ROC Curves on Training Data
<p align="center">
  <img src="images/m_roc_train.png" width = 400>
  <img src="images/p_roc_train.png" width = 400>
</p>

### ROC Curves on Test Data

<p align="center">
  <img src="images/m_roc_test.png" width = 400>
  <img src="images/p_roc_test.png" width = 400>
</p>

### Confusion Matrix
<p align="center">
  <img src="images/m_conf_mat.png" width = 400>
</p>



## Results and Interpretation


## Significant Coefficients



# Conclusion

[Back to Top](#Table-of-Contents)

# Citation
** 
[Back to Top](#Table-of-Contents)
