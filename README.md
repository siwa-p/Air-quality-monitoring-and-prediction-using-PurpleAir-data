# Air Quality: Machine learning models applied to air quality data

In this project, I have attempted to construct a predictive model for air-quality monitoring.
The particulate data (PM2.5) were obtained from the [OpenAQ](https://openaq.org) API for sensors in the Philadelphia Metropolitan area.
Various meteorological data were obtained from [NOAA](https://www.noaa.gov)

## Motivation

Air pollution data collected by the low-cost sensors are more useful in
applications including research, policy-making, public warnings, and
community education. Mostly because they have a denser presence
especially in urban areas. For example [PurpleAir](https://www2.purpleair.com/) has a global network of
over 13000 sensors.
While they tend to be less accurate, EPA has recently published a
correction schemes [1](https://doi.org/10.3390%2Fs22249669) to improve the comparability of these sensors data.
These kinds of low-cost sensors fills the spatial and temporal gaps in air-quality detection and provides valuable information that is easily accessible to public.

## Data Question

Can we use machine learning models to make reasonable air quality predictions
based on air-quality data from low-cost sensors along with other relevant
meteorological data (Rain, Snow, Wind, Temperature, Season, smoke
events, etc)?

Can we use neural networks as yet another method of learning to predict from such data?

## Collect data

The [OpenAQ v3 API](https://api.openaq.org/v3) is the primary data source, fetching PM2.5 sensor locations and daily/hourly measurements via `src/ingestion/get_openaq.py`. A free API key from [OpenAQ](https://explore.openaq.org/register) is required.

All sensor data is stored in **DuckDB** for efficient querying and analysis, replacing the earlier SQLite approach.

The weather data were downloaded from [NOAA](https://noaa.gov) via `src/ingestion/get_weather.py` for the needed dates and region. Since the region of our data is not so large, we did not see large variations in the weather data within the sensors of the region. So, weather data from a single meteorological station is used for analysis.

> **Note:** The analysis and model results below are based on the original Dallas-area PurpleAir dataset. Data ingestion for the Philadelphia OpenAQ dataset is currently in progress — results will be updated once ingestion completes and models are re-run against the new data.

## Preliminary EDA and thoughts

A quick look at the time-series data of a chosen sensor shows that the data is quite noisy with significant day to day variations. 

![time_series_random_sensor](assets/image.png)

### Time series Forecasting

#### Stationary?

ADF test is used to determine the presence of a unit root in a time series data and helps understand if the series is stationary or not.

Null Hypothesis: The series is non-stationary. 
Based on the result obtained for qute a few randomly chosen sensors: Here's result for one such instance

Results of Dickey-Fuller Test:

| Metric                        | Value          |
|-------------------------------|----------------|
| Test Statistic                | -8.192365e+00  |
| p-value                       | 7.607350e-13   |
| #Lags Used                    | 0.000000e+00   |
| Number of Observations Used   | 171.000000e+02 |
| Critical Value (1%)           | -3.469181e+00  |
| Critical Value (5%)           | -2.878595e+00  |
| Critical Value (10%)          | -2.575863e+00  |

The p-value is extremely low. Evidence for time series being stationary. 

#### ARMA model?

A quick look at the correlation in the time series data in the Figure below:

![figure](assets/image-1.png)

Both PACF and ACF plots show a lag of 1. A lag of 1 in PACF suggests that the future value is mostly determined by value one step behind in the time series (which is a day here). So a AR(1) model is appropriate.

While the lag of 1 in ACF is related to how the errors in the measurements today affects the measurements tomorrow. Hence a MA(1) becomes appropriate. 

With these results, We proceed with an ARIMA model with (1 0 1). We used SARIMAX from statsmodels, which includes in addition to the AR and MA, seasonality as well as exogenous variables from the data. The ARIMA implementation can be found [here](src/models/arima.py).
A quick look at the predictions on randomly chosen sensors due to ARIMA as explored in [this](notebooks/spatio_temporal.ipynb) is shown below.
![arima_predictions](assets/image-3.png)
The RMSE errors are added to the predictions. They are significantly larger than the day-to-day variations in the raw data. However, they do pick up the time-dependence. 

#### Caviat of ARIMA model:

Time series forecasting ignores the spatial dependence of the air quality data. The measurements of particulates in air is a local variable. Hence, it's prediction should include the information of the geo-location of the sensor in question

### Tree-based Regression Methods:

We will approach this in two ways:

- A temporal regression [temporal.py](src/models/temporal.py):

  - Tree based regression with features that correspond to a time series.

  - A lag feature was created for each sensor based on data of the day before. (.shift() method of pandas dataframes are handy)

  - XGBOOST is the choice of regression and produced the best results. The train-test split is performed ahead of time based on time series indexing which prevents data leakage.

- A spatial regression [spatial.py](src/models/spatial.py):

  - Tree based regression with features that capture spatial correlation

  - To generate spatial features, a weight matrix is generated based on inverse distance. Essentially, closer sensors are given larger weights.

  - Again XGBOOST regressor is used. The model was trained on all the sensors except the one. This was done for all the sensors. These were separate instances of the model. Hence, we do not expect data leakage to occur.

- Ensemble

  - Finally a simple ensemble (prediction averaged of the spatial and temporal) was constructed.

    - Predictions due to ensemble method (xgboost for spatial and time-series regressions) are a significant improvement over ARIMA model. These are explored in [this](/notebooks/spatio_temporal.ipynb) notebook. As seen below for a randomly chosen four sensors,

    ![ensemble_predicted](assets/image-4.png)

    RMSE errors are smaller and the predictions follow the data much closer. 

    Here's a distribution of prediction errors for the two methods

    ![scatterplot_predictions](assets/image-5.png)

    Both methods underestimate extreme values of observed pm2.5. Ensemble method does a much better job of staying closer to the observed values.


### Neural network

In order to capture both the spatial and time-series nature of the data, we are implementing a 3D convolutional neural network. A group of images (sequential in time) are stacked and fed into a convolutional neural network. An output is generated which is then compared (MSELoss) against the next image in sequence.

The data is a single channel (value of pm2.5) image of some dimensions. These data are stacked in timesteps of 10.

The deep network consists of block with 3D convolution, ReLU layers, batch normalization and skip connections to prevent gradient loss. The model is designed such that the image dimension is unchanged for prediction. (upconvolutions to the rescue)

The data loading and training is presented here [train](src/models/train_cnn.py).

- Data generation:
  - First a spherical earth is assumed and the latitude and longitude are converted into cartesian coordinates. The x and y coordinates are then taken to flatten the space. No projections done. 
  - Because the sensor locations are sparse with large areas unsampled, Kriging interpolations were performed based on a spherical Variogram model (`src/preprocessing/spatial_interpolation.py`). Here's a sample interpolated image for some point in time:

    ![Kriging](assets/image-2.png)

    The regions away from the data points generally have values closer to mean and larger variances.
- Once the images are generated, we can feed into the neural network, optimize the hyperparamters and train the model. 

- After training, a few test images are sampled from the dataloader and their predictions are presented side by side:

  ![cnn_predictions](assets/cnn_predictions.png)

  Looks like the model is learning from the images and can potentially predict values at some locations at some time in the future if a series of prior data is available.

### Spatio-Temporal Graph Neural Network

A Spatio-Temporal GNN (`STGNN`) is implemented in [src/models/gnn_model.py](src/models/gnn_model.py) and trained via [src/models/train_gnn.py](src/models/train_gnn.py). This approach models sensors as graph nodes with edges built by k-NN on lat/lon coordinates ([src/preprocessing/build_graph.py](src/preprocessing/build_graph.py)).

The architecture is:
- **Graph Attention Network (GATConv)** applied independently at each timestep to capture spatial dependencies between neighboring sensors
- **GRU** unrolled across timesteps to capture temporal dynamics
- A linear output head that produces the next-step PM2.5 prediction per node

Input is a tensor of shape `[T, N, F]` (timesteps × nodes × features) and output is `[N, 1]`, one prediction per sensor. EMA smoothing (`src/preprocessing/sensor_filter.py`) is applied to the raw PM2.5 signal before training to reduce noise.
