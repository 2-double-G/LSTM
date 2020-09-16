# How to use neural network LSTM with your data (google colab)
1.  **Upload datasets**

    Put your datasets in folder 'data' on your google disk or write filepath in function ```preprocess```:
    ```python
    path = '/content/drive/My Drive/data'
    ``` 
2.  **Data pre-processing**

    To preprocess your data write the filenames in list ```names``` in function ```preprocess```:
    ```python
    names = ['P1', 'P2', 'T11', 'T1', 'T21', 'Tnv']
    ``` 
    To read preprocessed data write filepath in function ```main```:
    ```python
    raw_data = read_data('/content/drive/My Drive/data/preprocessed.csv')
    ``` 
3. **Forecasting**

    In function ```main``` comment out this block of code if you don't need it:
    ```python
    # merge by hour and average:
    hours = raw_data.groupby(by=lambda dt: dt.floor('h')).mean()
    ``` 
    Choosing features ```X``` and the prediction target ```y```:
    ```python
    # choosing the features that we will use:
    X, y = np.array(hours[['Tnv']]), np.array(hours[['Tnv']])
    ``` 
    Select prediction parameters:
     ```python
    # forecast parameters: forecast 12 hours ahead with 48 hours of observations
    look_back, look_forward = 48, 12
    ``` 
    To split the data, choose the number of neurons, activation function, loss function, optimizer, epochs, and batch size you want:
    ```python
    X_val, X_train, y_val, y_train = train_test_split(X, y, test_size=0.90, shuffle=False)
    model = BasicModel(look_back=look_back,
                       look_forward=look_forward,
                       num_neurons_1=64,
                       # num_neurons_2=32,
                       # activation_2='relu',
                       activation_3='tanh',
                       loss='mean_squared_error',
                       optimizer='adam',
                       num_epochs=35,
                       batch_size=10)
    ``` 
