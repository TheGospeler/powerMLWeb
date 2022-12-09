# import libraries
import altair as alt
import hiplot as hip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Machine learning libraries and their estimators
import tensorflow as tf
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# scaling library
from sklearn.preprocessing import MinMaxScaler

# setting the layout for the seaborn plot
sns.set(style="darkgrid")

# The story I want to tell is the interaction or relationship between the power
# consumption in hot weather or cold climate.

# load DATA:- The data is loaded from the UCI machine learning Repository
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00616/Tetuan%20City%20power%20consumption.csv"
# data = pd.read_csv(url)

# Reading the data from the directory
file = r'data/Tetuan city power consumption.csv'
data = pd.read_csv(file)

# Enabling Time Series Analysis on data

data_RS = pd.read_csv(file, parse_dates=['DateTime'], index_col=['DateTime'])

# The dataset can only perform regressional tasks because most of the columns
# contains continuous data. In order to play with the data more I added a 
# classification column by creating a function based on the temperature ranges.

def warmth_condition(df):
    if df['Temperature'] >= 30:
        return '4-Hot'
    
    elif df['Temperature'] >= 20:  
        return '3-Mild'
    
    elif df['Temperature'] >= 10:
        return '2-Cold'
    
    return '1-Very Cold'

data['Warmth Level'] = data.apply(warmth_condition, axis=1)


# ------------------------------------------------------------------------------- #

# Functions for the Machine Learning Algorithms
def create_data(dataTS, columns, label):

    data = dataTS[columns]
    label = dataTS[label]

    # convert the data into series
    data = data.to_numpy()
    label = label.to_numpy()

    return data, label


def run_ML(data, label, est, scale=None, leaf=None, n_est=None, max_featD=None, max_featR=None):
    """"Run Machine Learning Model."""

    # Scaling
    if scale is not None:
        # We are going to make two instances for the scaling library
        data_scaler = MinMaxScaler(feature_range= scale)
        label_scaler  = MinMaxScaler(feature_range=scale)

        data = data_scaler.fit_transform(data)
        label = label_scaler.fit_transform(label.reshape(-1, 1))

    # This will be kept outside to select the ML we want to use in the prediction
    # For now we will use decision Tree

    if est == 'DT':

        rng = np.random.RandomState(2)
        #regr = AdaBoostRegressor(
            #DecisionTreeRegressor(min_samples_leaf=leaf, max_features=max_featD), n_estimators=300, random_state=rng)
        regr = DecisionTreeRegressor(min_samples_leaf=leaf, max_features=max_featD)
        # fit the data and label
        regr.fit(data, label)

        # Make prediction of the model
        y_pred = regr.predict(data)

    elif est == 'RF':
        regr = RandomForestRegressor(n_estimators=n_est, max_features=max_featR)

        # fit the data and label
        regr.fit(data, label)

        # Make prediction of the model
        y_pred = regr.predict(data)
        
    elif est == 'LR':
        regr = LinearRegression()

        # fit the data and label
        regr.fit(data, label)

        # Make prediction of the model
        y_pred = regr.predict(data)

        y_pred = np.ravel(y_pred)

    elif est == 'NN':
        linear_model = tf.keras.Sequential([
            tf.keras.Input(shape=(data.shape[1],)),  # Input layer; data=the features used for training
            Dense(units=25, activation=act, name='L1'),
            Dense(units=1, name='L2'),  # output layer
            ], name = "Neural_Estimator_Model")

        # configuring the function we want to use to fit the feature with the response (or label)
        linear_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss = 'mean_absolute_error')
        
        # Now, fitting the Model
        fitted_model = linear_model.fit(
            data, 
            label,
            epochs=epc,
            validation_split = 0.2
        )

        # Make prediction of the model
        y_pred = linear_model.predict(data)
        y_pred = np.ravel(y_pred)  # converts it to 1D array.
        


    # Invert the scaling
    if scale is not None:
        y_pred = label_scaler.inverse_transform([y_pred])[0]

    if est != 'NN':
        return y_pred, regr

    else:
        return y_pred, linear_model  # We used another predictor for the neural network

# Function that calculates the MSE and RMSE
def score_model(pred, actual):
    '''Calculate how the model performed.'''

    N = pred.shape[0]  # gets the number of the data
    mse =  sum((pred - actual)**2) / N
    rmse = np.sqrt(mse)

    return mse, rmse


# Creating New Features! Might turn this to a class, but for now let's use functions
def create_newFeat(data, feat_num):

    N = data.shape[0]    
    # arrays and vectors to store the new data
    x_data = np.zeros((N-feat_num, feat_num))
    yVec = np.zeros(N-feat_num)

    # loop through the data to extract the features
    for i in range(N-feat_num):
        for j in range(feat_num):
            x_data[i, j] = data[i+j]
            yVec[i] = data[i+feat_num]
    return x_data, yVec


def forcast_data(data, feat_num, forcast_size, regr):
    '''Forcast the rest of the data and see how the ML Learns.
    
    Params:
    data: The data is the last row of the training data.
    label: The label is the prediction based off the data.
    '''
    
    forcast = np.zeros(forcast_size)
    
    # Iterating and creating new data to make predictions
    for i in range(forcast_size):
        # extract the first feat_num digits and move
        to_predict = data[i:i+feat_num].reshape(1, feat_num)
        forcast[i] = regr.predict(to_predict)
        
        # Add the next predicted data to the data
        data = np.append(data, forcast[i])
        
    return forcast


# Designing the Visuals on the App
# --------------------------------

# Partitioning the Web App to accommodate the Visualization of the Dataset and ML algorithm
st.sidebar.write("""
    ### The Functionality of this App is divided into two sections""")

main_opt = st.sidebar.radio('Targeted Action: ', ["Run Machine Learning Algorithms", "Data Description and Visualization"])

# Courtesy: Write Something Fancy about yourself


if(main_opt == "Data Description and Visualization"):

    # The title of the web page
    st.write("""
    ## Do Weather Influence the rate of Power Consumption? 
    Using the Tetouan City Power Consumption Dataset as a case study.
    """)

    # show the dataset description
    if st.checkbox('Show written description of the Dataset'):
        st.write("""
    The Dataset originally constists of Nine columns namely: DateTime, temperature (°C), humidity, wind speed of Tetouan City, power supply 
    (general diffuse flows and diffuse flows) and the consumption for three different locations in the City (Zone 1, Zone 2, and Zone 3). 

    An extra column (Warmth Level) was added to the dataset to aid categorizing the data. The values of each feature (column) were collected 
    after every 10 minutes leading to a total of 52416 observations present in the data.
    """)



    zone_option = st.selectbox(
        "The values shows the maximum (Upper) and minimum (Lower) for the selected columns.  Select Zone",
        ('Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption')) 

    # Displaying a basic summary of the dataset
    # 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{data['Temperature'].max():.2f} ", f"{data['Temperature'].min():.2f}", delta_color="off")
    col2.metric("Humidity", f"{data['Humidity'].max():.2f}", f"{data['Humidity'].min():.2f}", delta_color="off")
    col3.metric("Wind Speed", f"{data['Wind Speed'].max():.2f}", f"{data['Wind Speed'].min():.2f}", delta_color="off")
    col4.metric(zone_option, f"{data[zone_option].max():.2f}", f"{data[zone_option].min():.2f}", delta_color="off")


    # Constraining the Data to output the values of the corresponding columns based on the selected column
    constrain_option = st.selectbox(
        "Constrain the corresponding Max and Min values based on the selected Column in the dataset.",
        ('Temperature', 'Humidity', 'Wind Speed', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption')) 

    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    ccol1.metric("Temperature", f"{data['Temperature'].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f} ", 
        f"{data['Temperature'].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")
    ccol2.metric("Humidity", f"{data['Humidity'].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f}", 
        f"{data['Humidity'].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")
    ccol3.metric("Wind Speed", f"{data['Wind Speed'].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f}", 
        f"{data['Wind Speed'].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")
    ccol4.metric(zone_option, f"{data[zone_option].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f}", 
        f"{data[zone_option].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")


    # Button that allows the user to see the entire table
    if st.checkbox('Show the Dataset'):
        st.dataframe(data=data)


    # Plotting some visuals
    st.write(
        "#### Resampling provides a wide scale of exploration of the Dataset")

    # resampling the data demands taking the mean of the dataset given the number of How
    hour = st.slider('Select the number of hours (e.g.,  1: Hourly,  24: Daily,  168: Weekly,  730: Monthly)', 0, 731, 1)

    # resampling the Tetouan DataFrame.
    if hour != 0:
        df = data_RS.resample(f'{hour}H').mean()  # Calculates the mean of the group
        df.reset_index(inplace=True)  # Resets the index to the original so that DateTime can be a column in the dateaframe
        df['Warmth Level'] = df.apply(warmth_condition, axis=1)  # Adds the Warmth level' column to the newly creaeted dataframe.



    if st.checkbox('Show the Resampled Dataset'):
        if hour == 0:
            st.write("""The data is not resampled, you must select atleast 1 on the slide bar!""")

        else:
            st.dataframe(data=df)


    # Select which of the dataframe to plot either the resampled or the actual dataset based on the value of the hour
    if hour == 0:
        df_plot = data  # saves the original plot to be plotted below
        
    else:
        df_plot = df  # saves the resampled plot to be plotted below



    # plotting Capabilities.
    st.write("""
        ### Select the different options to visualize the dataset below.
        """)

    # Plotting options
    plot_opt = st.selectbox("",
        ('Altair Interactive Plot', 'HiPlot', 'Joint Plot')) 

    # Selecting the different options for plotting

    if(plot_opt == 'Altair Interactive Plot'):

        opt1, opt2, opt3 = st.columns(3)
        df_sidebar = df_plot.drop(columns="Warmth Level")  # I don't want 'Warmth Level' to be in the options.
        with opt1:
            x_sb = st.selectbox('x axis: ', df_sidebar.columns)

        with opt2:
            y_sb = st.selectbox('y axis: ', df_sidebar.drop(columns=[x_sb]).columns)

        with opt3:
            color = st.selectbox('hue: ', ["Warmth Level"])

        # Making some interactive plots

        alt.data_transformers.disable_max_rows()  # To enable altair plot more than 5000 rows

        domain = ['1-Very Cold', '2-Cold', '3-Mild', '4-Hot']
        range_ = ['navy', 'royalblue', 'deepskyblue', 'crimson']
        chart = alt.Chart(df_plot).mark_point().encode(
            alt.X(x_sb, title= f'{x_sb}'),
            alt.Y(y_sb, title=f'{y_sb}'),
            color=alt.Color('Warmth Level', scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['DateTime', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 
                         'diffuse flows', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 
                     'Zone 3  Power Consumption']
        ).properties(
            width=600,
            height=400
        ).interactive()

        # plotting Altair with streamlit
        st.altair_chart(chart)


    elif(plot_opt == 'HiPlot'):

        feat_opt = st.multiselect(
            'Add or remove features to visualize in the plot below',
            ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows',
            'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption'], 
            ['Temperature', 'Humidity', 'Wind Speed',  'Zone 1 Power Consumption'])

        # Incase the user makes a mistake by deleting the columns by mistake
        if(len(feat_opt) == 0):
            st.write("""
                ### You cannot leave the field empty, Please select one or more columns!
                """)

        else:
            # New Dataset created based on the column selection
            Hiplot = df_plot[feat_opt]

            # Plotting the data using Hiplot
            xp = hip.Experiment.from_dataframe(Hiplot)
            xp.to_streamlit(key="hip").display()


    elif(plot_opt == 'Joint Plot'):
        J_opt1, J_opt2 = st.columns(2)
        df_sidebar = df_plot.drop(columns=["Warmth Level", "DateTime"])  # I don't want 'Warmth Level' to be in the options.
        with J_opt1:
            x_jp = st.selectbox('x axis: ', df_sidebar.columns)

        with J_opt2:
            y_jp = st.selectbox('y axis: ', df_sidebar.drop(columns=[x_jp]).columns)

        # Plotting with the jointplot
        sns.jointplot(x=x_jp, y= y_jp, data =df_plot)

        # Displaying the Plot using the streamlit command
        st.pyplot(plt.gcf())


#  ---------------------------------------------------------------------------------------------- #
#  Machine Learning Part
#  ---------------------------------------------------------------------------------------------- #
else:
    st.write("""## MACHINE LEARNING APPLICATION""")
    tab1, tab2, tab3, tab4 = st.tabs(["Label prediction from Dataset", "Explanation of Time Series Analysis", 
        "Label Prediction from label", "Portfolio and Reference"])

    with tab1:
        st.write("""##### Select Machine Learning Algorithm (Estimator)""")

        alg = st.selectbox('Estimators: ', ['Decision Tree', 'Random Forest', 'Neural Network', 'Linear Regression'])

        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            features = st.multiselect('Select features:',
                ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows'], 
                ['Temperature', 'Humidity', 'Wind Speed'])

        with ml_col2:
            label_select = st.selectbox('Select label: ', ['Quad (Zone 1)', 'Smir (Zone 2)', 'Boussafou (Zone 3)',
                'Aggregate (average)'])

        # Just in case a mistake is made to cancel all the features in the selection
        max_ft = len(features)
        if max_ft == 0:
            st.write('''### Ooops! You have to pick atleast one feature dear 😉''')

        else:

            # copying the dataset to be used for the ML algorithm
            data_ml = data_RS.copy()

            # Adding the Scaling Capability to the dataset
            scale = None

            st.sidebar.write(''' --- ''')
            st.sidebar.write('''#### Feature Engineering
                ''')
            st.sidebar.write('''Scaling the data optimizes Machine Learning Algorithms.''')
            st.sidebar.write('''Select one scaling method:''')

            if st.sidebar.checkbox('Scale: Range(0, 1)'):
                scale = (0, 1)
            elif st.sidebar.checkbox('Scale: Range(-1, 1)'):
                scale = (-1, 1)

            # creating a flag for labels
            if label_select == 'Quad (Zone 1)':
                label = 'Zone 1 Power Consumption'
            elif label_select == 'Smir (Zone 2)':
                label = 'Zone 2  Power Consumption'
            elif label_select == 'Boussafou (Zone 3)':
                label = 'Zone 3  Power Consumption'
            else:
                # The aggregate is the average of the three location.
                data_ml['Aggregate'] = data_RS[['Zone 1 Power Consumption', 'Zone 2  Power Consumption',
                'Zone 3  Power Consumption']].mean(axis=1).to_numpy()

                label = 'Aggregate'


            # Conditioning the dataset for the Machine Learning
            data, label = create_data(data_ml, features, label)

            # Running the Individual Estimators
            if alg == 'Random Forest':

                ml_col_RF1, ml_col_RF2 = st.columns(2)
                with ml_col_RF1:
                    # Making a default value for the slider after rendering
                    if max_ft == 1:
                        max_featR = max_ft  # This will be used as the hyperparameter for Max Features
                    else: 
                        val = max_ft-1
                        max_featR = st.slider('Max Feature:', 1, max_ft, val, key='MLR1')

                with ml_col_RF2:
                    n_estR = st.slider('Number of Trees:', 20, 100, 50, key='MLR2')

                # Running the Random Forest Regressor Estimator
                y_pred, _ = run_ML(data, label, est='RF', scale=scale, n_est=n_estR, max_featR=max_featR)

            elif alg == 'Decision Tree':
                ml_col_DT1, ml_col_DT2= st.columns(2)
                with ml_col_DT1:
                    # Making a default value for the slider after rendering
                    if max_ft == 1:
                        max_featD = max_ft  # This will be used as the hyperparameter for Max Features
                    else: 
                        val = max_ft-1
                        max_featD = st.slider('Max Feature:', 1, max_ft, val)

                with ml_col_DT2:
                    leaf_ML = st.slider('Number of Leafs:', 1, 10, 1, key='MLD1')

                # Running the Random Forest Regressor Estimator
                y_pred, _ = run_ML(data, label, est='DT', scale=scale, leaf=leaf_ML, max_featD=max_featD)

            elif alg == 'Linear Regression':
                # Running the Random Forest Regressor Estimator
                y_pred, _ = run_ML(data, label, est='LR', scale=scale)

            elif alg == 'Neural Network':
                ml_col_NN1, ml_col_NN2, ml_col_NN3= st.columns(3)
                with ml_col_NN1:
                    
                    # Selecting activation function
                    act = st.select_slider('Activation Function:', 
                        ['relu', 'selu', 'elu', 'gelu'])

                with ml_col_NN2:
                    lr = st.select_slider('Learning Rate:', [0.001, 0.01, 0.1])

                with ml_col_NN3:
                    epc = st.slider('Epoch:', 1, 100, 20)
                    epc = int(epc)

                # Running the Random Forest Regressor Estimator
                y_pred, _ = run_ML(data, label, est='NN', scale=scale)


            # Select the Range of Values to Plot
            st.subheader("Plotting The Actual and predicted trend")

            start, end = st.slider(
                'Select The Range of Values to plot', 
                0, data_ml.shape[0], [0, 1600])

            # Plotting the data
            plt.plot(label[start:end], label='Actual')
            plt.plot(y_pred[start:end], label='Predicted')
            plt.xlabel('Number of Observations')
            plt.ylabel(f'Power Consumption in {label_select}')
            plt.legend()


            # Displaying the Plot using the streamlit command
            st.pyplot(plt.gcf())

            # Displaying the Metrics of the Model
            mse, rmse = score_model(y_pred, label)

            st.write("""###### Model Performance:""")
            df = pd.DataFrame(np.array([mse, rmse]).reshape(1, -1), columns=('MSE', 'RMSE'))
            st.table(df)

    with tab2:
        st.write("""The Tetouan dataset is a time series data, and you must have observed the 
            seasonality in the dataset from the visualization on a small scale (10-minute intervals) 
            and large-scale (up to monthly intervals). From the observation of the dataset, 
            we see that the Power consumption in the three cities (zones) increases in the summer
            with some variabilities in comparison with the three cities. 
            In light of this, we can't split the data into train and test, not without
            doing a time series analysis on the data first (Think about it)! We cannot randomize the
            data, as we will mess up the trend. Also, If we split the dataset into train and 
            test, we will miss out on the trend of the test.

            """)

        st.write(""" The diagram below explains what the next section is doing. We split any of the
            desired labels into training and testing data in a sequential fashion, and then we create 
            a new set of features and label from the label dataset. We fit the estimator with the 
            training data and use the trained model to forecast the test data created with the same 
            approach.""")

        image = Image.open('data/image.png')
        st.image(image, "One-Step-Ahead Forecasts")

        st.write("""#### For Instance, 
            We consider the first 5 labels of Zone 1, using 2 step-size""")

        # creating three columns
        col_TSA1, col_TSA2, col_TSA3 = st.columns(3)

        # creating the data
        inst_data = np.array(data_RS['Zone 1 Power Consumption'])[:5]
        inst_feat, inst_label = create_newFeat(inst_data, 2)

        with col_TSA1:
            df_inst_data = pd.DataFrame(inst_data, columns=['Orignal Label'])
            st.table(df_inst_data)

        with col_TSA2:
            df_inst_feat = pd.DataFrame(inst_feat, columns=('Feature %d' % i for i in range(2)))
            st.table(df_inst_feat)

        with col_TSA3:
            df_inst_label = pd.DataFrame(inst_label, columns=['New Label'])
            st.table(df_inst_label)

        st.write("""**Note:** The next section allows you to Modify the column (or step) size of the feature and \
            see how the model and forcast behaves
            """)

    with tab3:
        # key="horizontal"
        st.session_state.horizontal = True

        # overright the scale to be None, the regressor scales the data, and I'm trying to figure out
        # inverting the scaling, once I am through, I will remove the scale = None
        scale = None

        label_LP = st.radio("Select Power Label:", 
            ['Quad (Zone 1)', 'Smir (Zone 2)', 'Boussafou (Zone 3)', 
            'Average (Aggregate)'], horizontal = st.session_state.horizontal)

        # create Dataset for this TSA
        data_TSA = data_RS.copy()[['Zone 1 Power Consumption', 'Zone 2  Power Consumption',
                        'Zone 3  Power Consumption']]

        # flags the label_LP
        # creating a flag for labels
        if label_LP == 'Quad (Zone 1)':
            labelTSA = 'Zone 1 Power Consumption'
        elif label_LP == 'Smir (Zone 2)':
            labelTSA = 'Zone 2  Power Consumption'
        elif label_LP == 'Boussafou (Zone 3)':
            labelTSA = 'Zone 3  Power Consumption'
        else:
            # The aggregate is the average of the three location.
            data_TSA['Aggregate'] = data_TSA.mean(axis=1).to_numpy()

            labelTSA = 'Aggregate'

        # Prepare the data label
        data_label = data_TSA[labelTSA]

        # Select Feature
        feat_select = st.slider('Choose the Number of step-size (Columns):', 1, 1000, 50)

        # creating the new feature
        features, label = create_newFeat(data_label, feat_select)  # Creating Data

        # Printing the shapes of the new data
        st.write("The shape of the new data is", features.shape, "and the shape of the label is",
            label.shape)

        # Split the Original Data
        split = st.slider('Split Data into Train - Test:', 0.1, 0.9, 0.8)

        # Splitting the dataset
        train_features = features[:int(split*features.shape[0])]
        train_label = label[:int(split*features.shape[0])]
        test_label = label[int(split*features.shape[0]):]
        # There's no need to prepare the test_features since we are going to forcast the labels.

        # Printing out the data dimension
        st.write("The size of the training data is", train_features.shape[0], "and the size of the test data is",
            test_label.shape[0])


        alg_TSA = st.selectbox('Select Estimator: ', ['Decision Tree', 'Random Forest', 'Linear Regression'])


        # Reusing the Code above
        max_ft = train_features.shape[1]

        # Running the Individual Estimators
        if alg_TSA == 'Random Forest':

            ml_col_TSA1, ml_col_TSA2 = st.columns(2)
            with ml_col_TSA1:
                # Making a default value for the slider after rendering
                if max_ft == 1:
                    max_featR = max_ft  # This will be used as the hyperparameter for Max Features
                else: 
                    val = int(max_ft/2)
                    max_featR = st.slider('Max Feature:', 1, max_ft, val, key='TSAR')

            with ml_col_TSA2:
                n_est_TSA = st.slider('Number of Trees:', 20, 100, 50, key='TSAN')

            # Running the Random Forest Regressor Estimator
            _, regression = run_ML(train_features, train_label, est='RF', scale=scale, n_est=n_est_TSA, max_featR=max_featR)
            y_forcast = forcast_data(train_features[-1, :], feat_select, test_label.shape[0], regression)

        elif alg_TSA == 'Decision Tree':
            ml_col_DT1, ml_col_DT2= st.columns(2)
            with ml_col_DT1:
                # Making a default value for the slider after rendering
                if max_ft == 1:
                    max_featD = max_ft  # This will be used as the hyperparameter for Max Features
                else: 
                    val = int(max_ft/2)
                    max_featD = st.slider('Max Feature:', 1, max_ft, val, key='TSAD')

            with ml_col_DT2:
                leaf_TSA = st.slider('Number of Leaf:', 1, 10, 1, key='TSAL')
                leaf = leaf_TSA

            # Running the Random Forest Regressor Estimator
            _, regression  = run_ML(train_features, train_label, est='DT', scale=scale, leaf=leaf_TSA, max_featD=max_featD)
            y_forcast = forcast_data(train_features[-1, :], feat_select, test_label.shape[0], regression)

        elif alg_TSA == 'Linear Regression':
            # Running the Random Forest Regressor Estimator
            _, regression  = run_ML(train_features, train_label, est='LR', scale=scale)
            y_forcast = forcast_data(train_features[-1, :], feat_select, test_label.shape[0], regression)

        elif alg_TSA == 'Neural Network':
            ml_col_NN1, ml_col_NN2, ml_col_NN3= st.columns(3)
            with ml_col_NN1:
                
                # Selecting activation function
                act = st.select_slider('Activation Function:', 
                    ['relu', 'selu', 'elu', 'gelu'])

            with ml_col_NN2:
                lr = st.select_slider('Learning Rate:', [0.001, 0.01, 0.1])

            with ml_col_NN3:
                epc = st.select_slider('Epoch:',
                    [1, 100], 20)

            # Running the Random Forest Regressor Estimator
            _, regression = run_ML(train_features, train_label, est='NN', scale=scale)

            # WE pick the last data of the training data and use it to make predictions
            y_forcast = forcast_data(train_features[-1, :], feat_select, test_label.shape[0], regression)


        # Remove this block of code after creating a class for the functions 

        # Select the Range of Values to Plot
        st.subheader("Plotting The Actual and predicted (Forecasted) trend")

        plt_tab1, plt_tab2 = st.tabs(["Slicing through the data (Closer Look)", 
            "Actual Plot showing all the data"])

        with plt_tab1:
            Mark = st.slider(
                'Select The Range of Values to plot: The same values apply R-L Train \
                and L-R Test and Prdict', 
                0, test_label.shape[0], 1600)

            # Extracting the training row
            tr_w = train_label.shape[0]

            # Plotting the data
            # Let's slice the plots
            T_start = tr_w-Mark

            plt.clf ()  # Clear off previous images
            plt.plot(np.arange(T_start,tr_w), train_label[T_start:tr_w+1], label='train')
            plt.plot(np.arange(tr_w, tr_w+Mark), test_label[:Mark], label='test')
            plt.plot(np.arange(tr_w, tr_w+Mark), y_forcast[:Mark], label='predict') 
            plt.xlabel('Number of Observations')
            plt.ylabel(f'Power Consumption in {label_LP}')
            plt.legend()

            # Displaying the Plot using the streamlit command
            st.pyplot(plt.gcf())
       

        with plt_tab2:

            # Extracting the training row
            tr_w = train_label.shape[0]
            te_w = test_label.shape[0]

            plt.clf ()  # Clear off previous images
            plt.plot(np.arange(tr_w), train_label, label='train')
            plt.plot(np.arange(tr_w, tr_w+te_w), test_label, label='test')
            plt.plot(np.arange(tr_w, tr_w+te_w), y_forcast, label='predict')  
            plt.xlabel('Number of Observaions')
            plt.ylabel(f'Power Consumption in {label_LP}')
            plt.legend()

            # Displaying the Plot using the streamlit command
            st.pyplot(plt.gcf())
        
    with tab4:
        # st.balloons()

        #  A brief portfolio
        port1, port2 = st.columns(2)

        with port2:
            st.write('''
                John Salako is currently a 
                 Research Assistant and a graduate student in Earth and Environmental Sciences, 
                Michigan State University (MSU).  John’s Interest in Data Science and Machine Learning
                Started in 2020 when he enrolled in New Horizons Computer Training Center to learn
                Python Programming. 

                He further completed the Worldquant University Data Science Module I 
                and II with honors, earning Credly Badges. His passion for knowledge led him to complete 
                additional certificated courses in LinkedIn, Coursera, and Udacity, taught by experts from 
                Google, MIT, and other institutions.

                ''')

        with port1:
            st.image('data/John.jpg')

        st.markdown('''
            This project is a class project of CMSE 830 (Foundations of Data Science), taught by a 
            stupendous Professor, Dr. Michael Murillo. 
            Dr. Murillo gave me the idea used in the Time Series Analysis.


            Feel Free to Connect via [LinkedIn](https://www.linkedin.com/in/john-salako/). I am looking forward 
            to possible collaboration and networking. You can view some of my previous works found in 
            my [GitHub Account](https://github.com/TheGospeler)


            ### Reference
            - Notes From Dr. Murillo (Image used for the Explanation)
            - The work of Salam et al. (2018) was used as a reference to this project.
                - [Comparison of Machine Learning Algorithms for the Power Consumption Prediction Case Study of Tetouan city](https://ieeexplore.ieee.org/document/8703007)

            ''')

