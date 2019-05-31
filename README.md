
# Module 3 Final Project
----------------
## Predict Medical Appointment No Shows
Why do 30% of patients miss their scheduled appointments?

### 2019.05.31

### Hussein Sajid & Anna Zubova 

[Jupyter Notebook](https://github.com/AnnaLara/mod_3_project_classification/blob/hussein/mod3/Index%20Part%201.ipynb)

[Slides](https://docs.google.com/presentation/d/1f0TGXD4iM-Tzlq4XLiDlYQ1kuPP1vhl447IFY7emRrQ/edit?usp=sharing)

**Project Goal**

The goal of this project is to test our ability to gather information from a real-world database and use our knowledge of statistical analysis and classification to generate analytical insights, build and interpret a classification model that can be meaningful to the company/stakeholder.

**Classification Model Requirements**

The goal of our project is to query the database to get the data needed to perform a statistical analysis and build a classification models. In this classification model, we will need to apply different classifier on the different models to answer at least one of the questions from the dataset we choose. 

For each classification model, be sure to specify the training set score and accuracy score of each classifier. 

## Dataset: Medical Appointments No Show
Found [here (link to kaggle.com)](https://www.kaggle.com/joniarroba/noshowappointments)

### Context
A person makes a doctor appointment, receives all the instructions and no-show. Who to blame? 

### Content
300k medical appointments and its 15 variables (characteristics) of each. The most important one if the patient show-up or no-show the appointment.

### Preliminary EDA

We first did quick EDA on the variables/features in the dataset to get familiar with them and to identify any that might need extra cleaning.

Here are the list of features in the dataset:

| Data Columns             | Entries      |
|--------------------------|--------------|
| PatientId                | 110527       | 
| AppointmentID            | 110527       | 
| Gender                   | 110527       |
| ScheduledDay             | 110527       | 
| AppointmentDay           | 110527       | 
| Age                      | 110527       | 
| Neighbourhood            | 110527       | 
| Scholarship              | 110527       |
| Hipertension             | 110527       | 
| Diabetes                 | 110527       | 
| Alcoholism               | 110527       | 
| Handcap                  | 110527       | 
| SMS_received             | 110527       |
| No-show                  | 110527       | 

## Data Wrangling

Firstly, I am going to try and explore the data to check for missing values/erroneous entries and also comment on redundant features and add additional ones, if needed.

* Data types
* No null values
* Imbalanced classes of predicted variable: only 20% No-shows
* Transform variables into binary (one-hot-encoding, Gender(0/1), No-show(0/1))

It is immediately apparent that some of the column names have typos, so let us clear them up before continuing further, so that I don't have to use alternate spellings everytime I need a variable. For convenience, I am going to convert the AppointmentDate and ScheduleDate columns into datetime64 format. It is interesting to note that the time portions have vanished from the Appointment Data timedeltas, because all appointment times were set exactly to 00:00:00. We also create a new feature called HourOfTheDay, which will indicate the hour of the day at which the appointment was booked. This will be derived off AppointmentDate. It is clear that we do not have any NaNs anywhere in the data. However, we do have some impossible ages such as -2 and -1, and some pretty absurd ages such as 100 and beyond. I do admit that it is possible to live 113 years and celebrate living so long, and some people do live that long, but most people don't.

## Exploring The Data

Now we are all set to explore the different features of the data and determine how good a feature it is for prediction whether a patient is likely to show up at an appointment.
First we will check how the likelihood that a person will show up at an appointment changes with respect to Age, HourOfTheDay, AwaitingTime. Clearly, HourOfTheDay and AwaitingTime are not good predictors of Status, since the probability of showing up depends feebly on the HourOfTheDay and not at all on the AwaitingTime. The significantly stronger dependency is observed with respect to Age.

## Baseline Modelling

We are going to start modeling to learn more above our variables! For this first run we are going to use ALL our non-categorical variables.

Following are the list of classifier used to predict the model accuracy:

* Logistic Regression
* Random Forest
* Support Vector Classifier 
* Class Imbalacing
* Decision Tree

For us, among all the classifier Random Forest gives better model Accuracy (80 %)

| No-show    | Precision    | Recall    | F1-Score |
|------------|--------------|-----------|----------|
| No         | 0.80         | 0.99      | 0.89     |
| Yes        | 0.32         | 0.02      |0.03      | 


--------------------

## Introduction

In this lesson, we'll review all the guidelines and specifications for the final project for Module 3.


## Objectives

* Understand all required aspects of the Final Project for Module 3
* Understand all required deliverables
* Understand what constitutes a successful project

## Final Project Summary

Congratulations! You've made it through another _intense_ module, and now you're ready to show off your newfound Machine Learning skills!

<img src='https://raw.githubusercontent.com/cenuno/dsc-3-final-project/master/smart.gif'>

All that remains for Module 3 is to complete the final project!

## The Project

For this project, you're going to select a dataset of your choosing and create a classification model. You'll start by identifying a problem you can solve with classification, and then identify a dataset. You'll then use everything you've learned about Data Science and Machine Learning thus far to source a dataset, preprocess and explore it, and then build and interpret a classification model that answers your chosen question.


### Selecting a Data Set

We encourage you to be very thoughtful when identifying your problem and selecting your data set--an overscoped project goal or a poor data set can quickly bring an otherwise promising project to a grinding halt.

To help you select an appropriate data set for this project, we've set some guidelines:


1. Your dataset should work for classification. The classification task can be either binary or multi-categorical, as long as it's a classification model.   

2. Your dataset needs to be of sufficient complexity. Try to avoid picking an overly simple dataset. We want to see all the steps of the Data Science Process in this project--it's okay if the dataset is mostly clean, but we expect to see some preprocessing and exploration. See the following section, **_Data Set Constraints_**, for more information on this.   

3. On the other end of the spectrum, don't pick a problem that's too complex, either. Stick to problems that you have a clear idea of how you can use machine learning to solve it. For now, we recommend you stay away from overly complex problems in the domains of Natural Language Processing or Computer Vision--although those domains make use of Supervised Learning, they come with a lot of other special requirements and techniques that you don't know yet (but you'll learn soon!). If you're chosen problem feels like you've overscoped, then it probably is. If you aren't sure if your problem scope is appropriate, double check with your instructor!

4. **_Serious Bonus Points_** if some or all of the data is data you have to source yourself through web scraping or interacting with a 3rd party API! Having projects that show off your ability to source data effectively make you look that much more impressive when showing your work off to potential employers!

### Data Set Constraints

When selecting a data set, be sure to take into consideration the following constraints:

1. Your data set can't be one we've already worked with in the previous two projects. 
2. Your data set should contain a minimum of 1000 rows.    
3. Your data set should contain a minimum of 10 predictor columns, before any one-hot encoding is performed.   
4. Your instructor must provide final approval on your data set.

### Problem First, or Data First?

There are two ways that you can about getting started: **_Problem-First_** or **_Data-First_**. 

**_Problem-First_**: Start with a problem that you want to solve with classification, and then try to find the data you need to solve it. If you can't find any data to solve your problem, then you should pick another problem. 

**_Data-First_**: Take a look at some of the most popular internet repositories of cool data sets we've listed below. If you find a data set that's particularly interesting for you, then it's totally okay to build your problem around that data set. 

There are plenty of amazing places that you can get your data from. We recommend you start looking at data sets in some of these resources first:

* [UCI Machine Learning Datasets Repository](https://archive.ics.uci.edu/ml/datasets.php)
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [Awesome Datasets Repo on Github](https://github.com/awesomedata/awesome-public-datasets)
* [New York City Open Data Portal](https://opendata.cityofnewyork.us/)
* [Seattle Open Data Portal](https://data.seattle.gov/)
* [Chicago Open Data Portal](https://data.cityofchicago.org/)
* [Inside AirBNB ](http://insideairbnb.com/)


## The Deliverables

There will be 2 deliverables for this project:

1. A well documented **Jupyter Notebook** containing any code you've written for this project and comments explaining it. This work will need to be pushed to your GitHub repository in order to submit your project.   

2. An executive **Keynote/PowerPoint/Google Slides presentation** (delivered as a PDF export) that explains the business problem you are solving, your findings, and their relevance to the company/stakeholders.
    + Contain between 5-10 professional quality slides detailing:
        + A high-level overview of your business problem, methodology and data source;
        + Any real-world recommendations you would like to make based on your findings (ask yourself--why should the executive team care about what you found? How can your findings help the company/stakeholder?);
        + Take no more than 10 minutes to present; and
        + Avoid technical jargon and explain results in a clear, actionable way for non-technical audiences.
    + Make sure to also add and commit this pdf of your non-technical presentation to your repository with a file name of `presentation.pdf`.


### Jupyter Notebook Must-Haves

For this project, your jupyter notebook should meet the following specifications:

**_Organization/Code Cleanliness_**

* The notebook should be well organized, easy to follow, and code is commented where appropriate.  
    * Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code. All functions have docstrings that act as professional-quality documentation.  
* The notebook is written to technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.

**_Process, Methodology, and Findings_**

* Your notebook should contain a clear record of your process and methodology for exploring and preprocessing your data, building and tuning a model, and interpreting your results.
* We recommend you use the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) process to help organize your thoughts and stay on track.


## Submitting your Project

You’re almost done! In order to submit your project for review, include the following links to your work in the corresponding fields on the right-hand side of Learn.

1. **GitHub Repo:** Now that you’ve completed your project in Jupyter Notebooks, push your work to GitHub and paste that link to the right. (If you need help doing so, review the resources [here](https://docs.google.com/spreadsheets/d/1CNGDhjcQZDRx2sWByd2v-mgUOjy13Cd_hQYVXPuzEDE/edit#gid=0).)
2. **Keynote/PowerPoint/Google Slides presentation:** be sure to commit a pdf of your non-technical presentation to the repository with a file name of `presentation.pdf`.

