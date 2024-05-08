# Big G Express: Predicting Derates
In this project, you will be working with fault code data and vehicle onboard diagnostic data to try and predict an upcoming full derate. These are indicated by an SPN 5246. 

You have been provided with a two files containing the data you will use to make these predictions (J1939Faults.csv and VehicleDiagnosticOnboardData.csv) as well as two files describing some of the contents (DataInfo.docx and Service Fault Codes_1_0_0_167.xlsx) 

Note that in its raw form the data does not have "labels", so you must define what labels you are going to use and create those labels in your dataset. Also, you will likely need to perform some significant feature engineering in order to build an accurate predictor.

There are service locations at (36.0666667, -86.4347222), (35.5883333, -86.4438888), and (36.1950, -83.174722), so you should remove any records in the vicinity of these locations, as fault codes may be tripped when working on the vehicles.

When evaluating the performance of your model, assume that the cost associated with a missed full derate is approximately $4000 in towing and repairs, and the cost of a false positive prediction is about $500 due to having the truck off the road and serviced unnecessarily.