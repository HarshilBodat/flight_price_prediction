#!/usr/bin/env python
# coding: utf-8

# # Flight Price Prediction
# ---

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set()
import streamlit as st
def trainmodel():
    # ## Importing dataset
    #
    # 1. Since data is in form of excel file we have to use pandas read_excel to load the data
    # 2. After loading it is important to check the complete information of data as it can indication many of the hidden infomation such as null values in a column or a row
    # 3. Check whether any null values are there or not. if it is present then following can be done,
    #     1. Imputing data using Imputation method in sklearn
    #     2. Filling NaN values with mean, median and mode using fillna() method
    # 4. Describe data --> which can give statistical analysis

    # In[2]:

    train_data = pd.read_excel("Data_Train.xlsx")

    # In[3]:

    pd.set_option('display.max_columns', None)

    # In[4]:

    train_data.head()

    # In[5]:

    train_data.info()

    # In[6]:

    train_data["Duration"].value_counts()

    # In[7]:

    train_data.dropna(inplace=True)

    # In[8]:

    train_data.isnull().sum()

    # ---

    # ## EDA

    # From description we can see that Date_of_Journey is a object data type,\
    # Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction
    #
    # For this we require pandas **to_datetime** to convert object data type to datetime dtype.
    #
    # <span style="color: red;">**.dt.day method will extract only day of that date**</span>\
    # <span style="color: red;">**.dt.month method will extract only month of that date**</span>

    # In[9]:

    train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day

    # In[10]:

    train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.month

    # In[11]:

    train_data.head()

    # In[12]:

    # Since we have converted Date_of_Journey column into integers, Now we can drop as it is of no use.

    train_data.drop(["Date_of_Journey"], axis=1, inplace=True)

    # In[13]:

    # Departure time is when a plane leaves the gate.
    # Similar to Date_of_Journey we can extract values from Dep_Time

    # Extracting Hours
    train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

    # Extracting Minutes
    train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute

    # Now we can drop Dep_Time as it is of no use
    train_data.drop(["Dep_Time"], axis=1, inplace=True)

    # In[14]:

    train_data.head()

    # In[15]:

    # Arrival time is when the plane pulls up to the gate.
    # Similar to Date_of_Journey we can extract values from Arrival_Time

    # Extracting Hours
    train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour

    # Extracting Minutes
    train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

    # Now we can drop Arrival_Time as it is of no use
    train_data.drop(["Arrival_Time"], axis=1, inplace=True)

    # In[16]:

    train_data.head()

    # In[17]:

    # Time taken by plane to reach destination is called Duration
    # It is the differnce betwwen Departure Time and Arrival time

    # Assigning and converting Duration column into list
    duration = list(train_data["Duration"])

    for i in range(len(duration)):
        if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
            else:
                duration[i] = "0h " + duration[i]  # Adds 0 hour

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

    # In[18]:

    # Adding duration_hours and duration_mins list to train_data dataframe

    train_data["Duration_hours"] = duration_hours
    train_data["Duration_mins"] = duration_mins

    # In[19]:

    train_data.drop(["Duration"], axis=1, inplace=True)

    # In[20]:

    train_data.head()

    # ---

    # ## Handling Categorical Data
    #
    # One can find many ways to handle categorical data. Some of them categorical data are,
    # 1. <span style="color: blue;">**Nominal data**</span> --> data are not in any order --> <span style="color: green;">**OneHotEncoder**</span> is used in this case
    # 2. <span style="color: blue;">**Ordinal data**</span> --> data are in order --> <span style="color: green;">**LabelEncoder**</span> is used in this case

    # In[21]:

    train_data["Airline"].value_counts()

    # In[22]:

    # From graph we can see that Jet Airways Business have the highest Price.
    # Apart from the first Airline almost all are having similar median

    # Airline vs Price
    sns.catplot(y="Price", x="Airline", data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6,
                aspect=3)
    plt.show()

    # In[23]:

    # As Airline is Nominal Categorical data we will perform OneHotEncoding

    Airline = train_data[["Airline"]]

    Airline = pd.get_dummies(Airline, drop_first=True)

    Airline.head()

    # In[24]:

    train_data["Source"].value_counts()

    # In[25]:

    # Source vs Price

    sns.catplot(y="Price", x="Source", data=train_data.sort_values("Price", ascending=False), kind="boxen", height=4,
                aspect=3)
    plt.show()

    # In[26]:

    # As Source is Nominal Categorical data we will perform OneHotEncoding

    Source = train_data[["Source"]]

    Source = pd.get_dummies(Source, drop_first=True)

    Source.head()

    # In[27]:

    train_data["Destination"].value_counts()

    # In[28]:

    # As Destination is Nominal Categorical data we will perform OneHotEncoding

    Destination = train_data[["Destination"]]

    Destination = pd.get_dummies(Destination, drop_first=True)

    Destination.head()

    # In[29]:

    train_data["Route"]

    # In[30]:

    # Additional_Info contains almost 80% no_info
    # Route and Total_Stops are related to each other

    train_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

    # In[31]:

    train_data["Total_Stops"].value_counts()

    # In[32]:

    # As this is case of Ordinal Categorical type we perform LabelEncoder
    # Here Values are assigned with corresponding keys

    train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

    # In[33]:

    train_data.head()

    # In[34]:

    # Concatenate dataframe --> train_data + Airline + Source + Destination

    data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)

    # In[35]:

    data_train.head()

    # In[36]:

    data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

    # In[37]:

    data_train.head()

    # In[38]:

    data_train.shape

    # ---

    # ## Test set

    # In[39]:

    test_data = pd.read_excel("Test_set.xlsx")

    # In[40]:

    test_data.head()

    # In[41]:

    # Preprocessing

    print("Test data Info")
    print("-" * 75)
    print(test_data.info())

    print()
    print()

    print("Null values :")
    print("-" * 75)
    test_data.dropna(inplace=True)
    print(test_data.isnull().sum())

    # EDA

    # Date_of_Journey
    test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
    test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format="%d/%m/%Y").dt.month
    test_data.drop(["Date_of_Journey"], axis=1, inplace=True)

    # Dep_Time
    test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
    test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
    test_data.drop(["Dep_Time"], axis=1, inplace=True)

    # Arrival_Time
    test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
    test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
    test_data.drop(["Arrival_Time"], axis=1, inplace=True)

    # Duration
    duration = list(test_data["Duration"])

    for i in range(len(duration)):
        if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
            else:
                duration[i] = "0h " + duration[i]  # Adds 0 hour

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

    # Adding Duration column to test set
    test_data["Duration_hours"] = duration_hours
    test_data["Duration_mins"] = duration_mins
    test_data.drop(["Duration"], axis=1, inplace=True)

    # Categorical data

    print("Airline")
    print("-" * 75)
    print(test_data["Airline"].value_counts())
    Airline = pd.get_dummies(test_data["Airline"], drop_first=True)

    print()

    print("Source")
    print("-" * 75)
    print(test_data["Source"].value_counts())
    Source = pd.get_dummies(test_data["Source"], drop_first=True)

    print()

    print("Destination")
    print("-" * 75)
    print(test_data["Destination"].value_counts())
    Destination = pd.get_dummies(test_data["Destination"], drop_first=True)

    # Additional_Info contains almost 80% no_info
    # Route and Total_Stops are related to each other
    test_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

    # Replacing Total_Stops
    test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

    # Concatenate dataframe --> test_data + Airline + Source + Destination
    data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)

    data_test.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

    print()
    print()

    print("Shape of test data : ", data_test.shape)

    # In[42]:

    data_test.head()

    # ---

    # ## Feature Selection
    #
    # Finding out the best feature which will contribute and have good relation with target variable.
    # Following are some of the feature selection methods,
    #
    #
    # 1. <span style="color: purple;">**heatmap**</span>
    # 2. <span style="color: purple;">**feature_importance_**</span>
    # 3. <span style="color: purple;">**SelectKBest**</span>

    # In[43]:

    data_train.shape

    # In[44]:

    data_train.columns

    # In[45]:

    X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
                           'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
                           'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                           'Airline_Jet Airways', 'Airline_Jet Airways Business',
                           'Airline_Multiple carriers',
                           'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                           'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                           'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                           'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
                           'Destination_Kolkata', 'Destination_New Delhi']]
    X.head()

    # In[46]:

    y = data_train.iloc[:, 1]
    y.head()

    # In[47]:

    # Finds correlation between Independent and dependent attributes

    plt.figure(figsize=(18, 18))
    sns.heatmap(train_data.corr(), annot=True, cmap="RdYlGn")

    plt.show()

    # In[48]:

    # Important feature using ExtraTreesRegressor

    from sklearn.ensemble import ExtraTreesRegressor
    selection = ExtraTreesRegressor()
    selection.fit(X, y)

    # In[49]:

    print(selection.feature_importances_)

    # In[50]:

    # plot graph of feature importances for better visualization

    plt.figure(figsize=(12, 8))
    feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()

    # ---

    # ## Fitting model using Random Forest
    #
    # 1. Split dataset into train and test set in order to prediction w.r.t X_test
    # 2. If needed do scaling of data
    #     * Scaling is not done in Random forest
    # 3. Import model
    # 4. Fit the data
    # 5. Predict w.r.t X_test
    # 6. In regression check **RSME** Score
    # 7. Plot graph

    # In[51]:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # In[52]:

    from sklearn.ensemble import RandomForestRegressor
    reg_rf = RandomForestRegressor()
    reg_rf.fit(X_train, y_train)

    # In[53]:

    y_pred = reg_rf.predict(X_test)

    # In[54]:

    reg_rf.score(X_train, y_train)

    # In[55]:

    reg_rf.score(X_test, y_test)

    # In[56]:

    sns.distplot(y_test - y_pred)
    plt.show()

    # In[57]:

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.show()

    # In[58]:

    from sklearn import metrics

    # In[59]:

    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # In[60]:

    # RMSE/(max(DV)-min(DV))

    2090.5509 / (max(y) - min(y))

    # In[61]:

    metrics.r2_score(y_test, y_pred)

    # In[ ]:

    # ---

    # ## Hyperparameter Tuning
    #
    #
    # * Choose following method for hyperparameter tuning
    #     1. **RandomizedSearchCV** --> Fast
    #     2. **GridSearchCV**
    # * Assign hyperparameters in form of dictionery
    # * Fit the model
    # * Check best paramters and best score

    # In[62]:

    from sklearn.model_selection import RandomizedSearchCV

    # In[63]:

    # Randomized Search CV

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]

    # In[64]:

    # Create the random grid

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    # In[65]:

    # Random search of parameters, using 5 fold cross validation,
    # search across 100 different combinations
    rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                                   n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

    # In[66]:

    rf_random.fit(X_train, y_train)

    # In[67]:

    rf_random.best_params_

    # In[68]:

    prediction = rf_random.predict(X_test)

    # In[69]:

    plt.figure(figsize=(8, 8))
    sns.distplot(y_test - prediction)
    plt.show()

    # In[70]:

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, prediction, alpha=0.5)
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.show()

    # In[71]:

    print('MAE:', metrics.mean_absolute_error(y_test, prediction))
    print('MSE:', metrics.mean_squared_error(y_test, prediction))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

    # ---
    file = open('flight_rf.pkl', 'wb')

    # dump information to that file
    pickle.dump(rf_random, file)
    # ## Save the model to reuse it again

    # In[77]:

def testmodel(airline,Source,Destination,Total_Stops,Depature_Date,Arrival_Date):
    import pickle
    # open a file, where you ant to store the data

    # In[78]:

    model = open('flight_rf.pkl', 'rb')
    forest = pickle.load(model)
    Journey_day = int(pd.to_datetime(Depature_Date, format="%Y-%m-%dT%H:%M").day)
    Journey_month = int(pd.to_datetime(Depature_Date, format="%Y-%m-%dT%H:%M").month)
    # print("Journey Date : ",Journey_day, Journey_month)

    # Departure
    Dep_hour = int(pd.to_datetime(Depature_Date, format="%Y-%m-%dT%H:%M").hour)
    Dep_min = int(pd.to_datetime(Depature_Date, format="%Y-%m-%dT%H:%M").minute)
    # print("Departure : ",Dep_hour, Dep_min)

    # Arrival
    Arrival_hour = int(pd.to_datetime(Arrival_Date, format="%Y-%m-%dT%H:%M").hour)
    Arrival_min = int(pd.to_datetime(Arrival_Date, format="%Y-%m-%dT%H:%M").minute)
    # print("Arrival : ", Arrival_hour, Arrival_min)

    # Duration
    dur_hour = abs(Arrival_hour - Dep_hour)
    dur_min = abs(Arrival_min - Dep_min)
    # print("Duration : ", dur_hour, dur_min)

    # Total Stops
    stop_map={"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3,"4 stops":4}
    Total_stops = stop_map.get(Total_Stops)
    # print(Total_stops)

    # Airline
    # AIR ASIA = 0 (not in column)
    if (airline == 'Jet Airways'):
        Jet_Airways = 1
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'IndiGo'):
        Jet_Airways = 0
        IndiGo = 1
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'Air India'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 1
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'Multiple carriers'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 1
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'SpiceJet'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 1
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'Vistara'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 1
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'GoAir'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 1
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'Multiple carriers Premium economy'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 1
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'Jet Airways Business'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 1
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline == 'Vistara Premium economy'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 1
        Trujet = 0

    elif (airline == 'Trujet'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 1

    else:
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    # print(Jet_Airways,
    #     IndiGo,
    #     Air_India,
    #     Multiple_carriers,
    #     SpiceJet,
    #     Vistara,
    #     GoAir,
    #     Multiple_carriers_Premium_economy,
    #     Jet_Airways_Business,
    #     Vistara_Premium_economy,
    #     Trujet)

    # Source
    # Banglore = 0 (not in column)
    if (Source == 'Delhi'):
        s_Delhi = 1
        s_Kolkata = 0
        s_Mumbai = 0
        s_Chennai = 0

    elif (Source == 'Kolkata'):
        s_Delhi = 0
        s_Kolkata = 1
        s_Mumbai = 0
        s_Chennai = 0

    elif (Source == 'Mumbai'):
        s_Delhi = 0
        s_Kolkata = 0
        s_Mumbai = 1
        s_Chennai = 0

    elif (Source == 'Chennai'):
        s_Delhi = 0
        s_Kolkata = 0
        s_Mumbai = 0
        s_Chennai = 1

    else:
        s_Delhi = 0
        s_Kolkata = 0
        s_Mumbai = 0
        s_Chennai = 0

    # print(s_Delhi,
    #     s_Kolkata,
    #     s_Mumbai,
    #     s_Chennai)

    # Destination
    # Banglore = 0 (not in column)
    if (Destination == 'Cochin'):
        d_Cochin = 1
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 0
        d_Kolkata = 0

    elif (Destination == 'Delhi'):
        d_Cochin = 0
        d_Delhi = 1
        d_New_Delhi = 0
        d_Hyderabad = 0
        d_Kolkata = 0

    elif (Destination == 'New_Delhi'):
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 1
        d_Hyderabad = 0
        d_Kolkata = 0

    elif (Destination == 'Hyderabad'):
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 1
        d_Kolkata = 0

    elif (Destination == 'Kolkata'):
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 0
        d_Kolkata = 1

    else:
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 0
        d_Kolkata=0


    y_prediction = forest.predict([[
            Total_stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            dur_hour,
            dur_min,
            Air_India,
            GoAir,
            IndiGo,
            Jet_Airways,
            Jet_Airways_Business,
            Multiple_carriers,
            Multiple_carriers_Premium_economy,
            SpiceJet,
            Trujet,
            Vistara,
            Vistara_Premium_economy,
            s_Chennai,
            s_Delhi,
            s_Kolkata,
            s_Mumbai,
            d_Cochin,
            d_Delhi,
            d_Hyderabad,
            d_Kolkata,
            d_New_Delhi
        ]])

    output=round(y_prediction[0],2)
    return output



st.title("Flight Fare Prediction")

# here we define some of the front end elements of the web page like
# the font and background color, the padding and the text to be displayed
html_temp = """
<div style ="background-color:lightpink;padding:13px">
<h1 style ="color:black;text-align:center;">Flight Fare App </h1>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html = True)
col1,col2,col3=st.columns(3)
airline = col1.selectbox('Select your Airline',('', 'Jet Airways', 'IndiGo','Air India','SpiceJet','Vistara','Air Asia','GoAir','Trujet','Multiple carriers Premium economy','Multiple carriers','Jet Airways Business','Vistara Premium economy'))
Source=col2.selectbox('Select your Departure City',('Delhi','Kolkata','Banglore','Mumbai','Chennai'))
Destination = col3.selectbox('Select your Destination City',('Cochin', 'Banglore', 'Delhi','New Delhi','Hyderabad','Kolkata'))

col4,col5,col6=st.columns(3)
Total_Stops = col4.selectbox('Select the No.of stops',('non-stop','1 stop','2 stops','3 stops','4 stops'))
Departure_Date=col5.date_input('Select your Date of Departure')
Arrival_Date=col6.date_input('Select your Date of Arrival')

col7,col8=st.columns(2)
result =""
if st.button("Predict"):
    model=trainmodel()
    result=testmodel(airline,Source,Destination,Total_Stops,Departure_Date,Arrival_Date)
    st.success('Suggested fare is: {}'.format(result))



