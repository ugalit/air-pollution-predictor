# air-pollution-predictor
Modeling atmospheric aerosol particles in Berlin based on different machine learning algorithms using weather, traffic and time features 

## Data

All datasets are publicly available.

+ Air quality: PM2.5 concentrations at measurement station in traffic (Mariendorfer Damm)
    
    https://luftdaten.berlin.de

+ Weather data: air temperature, humidity, wind speed, wind direction, precipitation (Alexanderplatz)
    https://opendata.dwd.de

+ Traffic data: quantity and velocity of cars per street (Mariendorfer Damm & distribution over other streets)

    https://api.viz.berlin.de/daten/verkehrsdetektion

<!-- + Time features: hour, weekday, month -->

## Models

Supervised learning | regression | time series

GridSearchCV for all models with K-fold cross-validation and hyperparameter tuning

- Linear Regression
- Linear regression with polynomial features
- Random Forest Regression
