Team: Anna Zubova, Hussein Sajid

Jupyter Notebook:

- [Part 1]()
- [Part 2](https://github.com/AnnaLara/mod_3_project_classification/blob/master/index_part_2.ipynb)

[Presentation slides](https://github.com/AnnaLara/mod_3_project_classification/blob/master/presentation.pdf)

We explored Kaggle dataset with data about medical appointments in Brazil during year 2016. The dataset contains data about approximately 110,000 appointments

Dataset [link](https://www.kaggle.com/joniarroba/noshowappointments)

Goal: predict a no-show using classification algorithms.

In part 1

## Part 2 

I part 2 we did some feature engineering to see if we could predict better the no-show comparing to the models from part 1.

The features we added:

- How many days in advance the appointment was made
- Appointment month
- Appointment day of the week
- Number of prior appointments for each appointment
- Number of prior no-shows for each appointment

We ran several classification algorithms on the data, here is a comparison of different algorithms' evaluation metrics:

| **model**  |  **cv score** |**f1 0 / 1**| **precision 0 / 1**  |**recall 0 / 1**   |
|---|---|---|---|---|
| decision trees  |  0.75 | 0.76 / 0.41  | 0.89 / 0.30  |  0.66 / 0.66 |
| random forests  | 0.71  | 0.87 / 0.25  | 0.83 / 0.32  |  0.90 / 0.20 |
|  logistic regression (initial) | 0.69  | 0.84 / 0.33  |  0.85 / 0.32 |  0.84 / 0.34 |
|  logistic regression (dropped features) | 0.69 | 0.84 / 0.33  |  0.85 / 0.32 |  0.83 / 0.35 |

Decision trees was one of the most successful algorithms, but it is prone to overfitting, so we applied other algorithms too.

Logistic regression gave us much more reasonable recall than models from part 2 with a bit of loss in Cross Validation score.

After applying logistic regression for the first time, we had to revome variables `Scholarship`, `Diabetes`, `month_appointment_5` and `day_of_week_appointment_5` because of the large p-values. Here is the remaining variables with corresponding coefficients:
 
Gender	-0.123217
Age	-0.007556
Hipertension	-0.007325
Alcoholism	0.040789
Handcap	-0.146144
SMS_received	0.390198
days_in_advance	0.027498
month_appointment_6	-0.463698
day_of_week_appointment_1	-0.244623
day_of_week_appointment_2	-0.175504
day_of_week_appointment_3	-0.245840
day_of_week_appointment_4	-0.096902
number_of_previous_apptms	-0.078217
number_of_previous_noshows	0.601255

### Interpretation of coefficients

Continious variables:

- **age**: for every additional year log(odds of no-show) decreases by 0.007
- **number_of_previous_apptms**: for 1 additional number log(odds of no-show) decreases by 0.07
- **days_in_advance**: for every additional day log(odds of no-show) increases by 0.027
- **number_of_previous_noshows** for every additional previous no-show log(odds of no-show) increases by 0.6

Discreet variables:

The presence of feature increases/decreases log(odds of no-show) by coefficient value

Most significant features: `number_of_previous_noshows`, `month_appointment_6`, `SMS_received` 

### Simulation of change in most significant variables

Let's predict the probability of a no-show for a person with the following parameters:

- Gender:	Male
- Age:	30
- Hipertension:	0
- Alcoholism: 	0
- Handicap:	    0
- SMS_received:	0
- days_in_advance:	14
- Month_appointment_6:  	             0
- Day_of_week_appointment_1: 	0
- Day_of_week_appointment_2: 	1
- Day_of_week_appointment_3: 	0
- Day_of_week_appointment_4: 	0
- Number_of_previous_apptms: 	0
- number_of_previous_noshows:	0

We used out Logistic Regression model to predict the probability of a no-show of this imaginary person. This gave us our baseline probability: 46%

We will now try to increase the parameters for our 3 most important variables to see how the predicted probability will be changing.

1. Number of previous no-shows:  increase from 0 to 1 
Increase of probability from 46% to 61%

2. Number of previous no-shows:  increase from 1 to 2 
Increase of probability from 61% to 74%

3. Appointment in June:
Decrease of probability from  74% to 64%

3. Received SMS notification:
Increase of probability from 62% to 72%

It is counter-intuitive that recieving SMS will increase the probability of a no-show. The description of the dataset states that "SMS_received = 1 or more messages sent to the patient". That could mean that the value 0 indicates that more than 1 SMS was sent for those patient, which might explain the positive relationship between a no-show and SMS_recieved variable.





