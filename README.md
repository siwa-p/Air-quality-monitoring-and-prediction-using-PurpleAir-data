# Air Quality: Machine learning models applied to air quality data
In this project, I will attempt to construct a predictive model for air-quality monitoring from data obtained from inexpensive air-sensors by PurpleAir and various meteorological data. I will use purpleAir API to gather data in a region(to be decided) for a period of time. Weather data will be obtained from BigQuery public database. 

An application will be build that will let users visualize the data, the model and it’s predictions. 

## Motivation:

Air pollution data collected by the low-cost sensors are more useful in
applications including research, policy-making, public warnings, and
community education. Mostly because they have a denser presence
especially in urban areas. For example [PurpleAir](https://www2.purpleair.com/) has a global network of
over 13000 sensors.
While they tend to be less accurate, EPA has recently published a
correction schemes [1](https://doi.org/10.3390%2Fs22249669) to improve the comparability of these sensors data.
These kinds of low-cost sensors fills the spatial and temporal gaps in air-
quality detection and provides valuable information that is easily accessible to public.

## Data Question
Can a machine learning model make reasonable air quality predictions
based on air-quality data from low-cost sensors along with other relevant
meteorological data (Rain, Snow, Wind, Temperature, Season, smoke
events, etc)?

## Minimum Viable Product
A streamlit dashboard for visualization of air quality data in city. Historical
air quality graphs.
A model to forecast air quality (Classification and Regression, Decision
Tree).
Artificial neural network trained on time series data for forecasting