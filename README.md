
# Health Condition Prediction App


## 1 - Introduction
This application is build to visualize the prediction result of the user with limited information. It utilizes the [disease prediction model](https://github.com/simon201918/Disease_Prediction_with_NHANES) that I developed in a previous project.

To run the App, please download all the files in this repository and do not change the name or path of any file. Then run the [app.py](https://github.com/simon201918/Health_Condition_Prediction_App/blob/main/app.py). Because Python will train the machine learning model each time you start the App, it may take about 30s to load the App fully. When Python is ready, open the browser with the link suggested by your IDE - most likely, the link is http://127.0.0.1:8050/. 

## 2 - App Instruction
When you open the browser, the App should look like this:

![say sth](https://github.com/simon201918/Health_Condition_Prediction_App/blob/main/img/img_1.png?raw=true)

All the values (except gender) have pre-defined values. When using the App, please fill in ALL the required information and then click the submit button on the screen's right side. 

The bar plot on the bottom-left corner will show your estimated probability of getting certain medical conditions. The result will change each time you update the information. 

The graph on the bottom-right corner displays the suggestion with your most likely condition so that you could learn how to prevent the disease.

Some screenshots of the App are listed below.

![say sth](https://github.com/simon201918/Health_Condition_Prediction_App/blob/main/img/img_2.png?raw=true)


![say sth](https://github.com/simon201918/Health_Condition_Prediction_App/blob/main/img/img_3.png?raw=true)


## 3 - Important Notice and Waiver of Liability

These machine learning models that power this App is trained with [***National Health and Nutrition Examination Survey (NHANES)*** ](https://www.cdc.gov/nchs/nhanes/index.htm?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fnchs%2Fnhanes.htm) data (2013-2014). For the details of this Survey and model tuning, please click [here](https://github.com/simon201918/Disease_Prediction_with_NHANES).

**All the suggestions provided by this application are only for your reference. This is not an official medical diagnose. The tips for disease prevention are downloaded from the internet and not backed by any doctor or hospital. I do NOT take ANY responsibility for the result and suggestion. Please always consult with your doctor about your health condition.**

If you believe any asset uses in this application is conflicted with your copyright, please contact me via email at simon201918@gmail.com.

Thanks for reading, and I hope you enjoy this App!
