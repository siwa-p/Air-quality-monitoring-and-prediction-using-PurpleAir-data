import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

predictors = ['humidity_a','temperature_a', 'pressure_a']#,
      # 'pm2.5_atm_a', 'pm2.5_atm_b']
target = ['pm2.5_cf_1_a']

class Model:
    def __init__(self, datafile, model):
        self.df = pd.read_csv(datafile)
        self.model = model()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
    def split(self, test_size, predictors, target):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[predictors], 
                                                                                self.df[target], 
                                                                                test_size=test_size, 
                                                                                random_state=321)
        self.X_train = self.pipeline.fit_transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)
    def fit(self):
        self.fit_model = self.model.fit(self.X_train, self.y_train)
        
    def predict(self):
        result = self.fit_model.predict(self.X_test)
        return result


def plot_data(model_instance, xlabel='xlabel', ylabel='ylabel', title='Plot', color='b', linestyle='-', marker=None, markersize=5):
    y_pred = model_instance.predict()
    # Calculate R^2 score
    r2_score = model_instance.fit_model.score(model_instance.X_test, model_instance.y_test)
    plt.figure(figsize=(8, 6)) 
    
    # Plot y=x line
    plt.plot(model_instance.y_test, model_instance.y_test, color='gray', linestyle='--', label='y=x')
    
    # Plot predicted vs true values
    plt.scatter(model_instance.y_test, y_pred, color=color, marker=marker, s=markersize, label='Data Points')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title + f' (R^2 Score: {r2_score:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    model_instance = Model('capstone/datasets/sensor_nashville.csv',LinearRegression)
    model_instance.split(0.2,predictors=predictors, target=target)
    model_instance.fit()
    
    #Print scores 
    
    # y_pred = model_instance.predict()
    # print(f'MSE: {mean_squared_error(model_instance.y_test, y_pred)}')
    # print(f'MAE: {mean_absolute_error(model_instance.y_test, y_pred)}')
    # print(f'R squared: {r2_score(model_instance.y_test, y_pred)}')
    
    # plot data 
    
    plot_data(model_instance, xlabel='True PM2.5 Values', 
              ylabel='Predicted PM2.5 Values', 
              title='True vs Predicted PM2.5 Values', 
              color='r', linestyle='-', marker='o', markersize=8)
    