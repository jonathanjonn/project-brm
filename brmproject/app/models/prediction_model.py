import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

class StockPredictor:
    def __init__(self):
        self.model = None
        self.history_data = None
        self.difference_order = 1  # Default differencing order
        self.seasonal_difference = 1  # Default seasonal differencing
        self.seasonal_period = 7  # Default seasonality (weekly)
        
    def prepare_data(self, transactions, product_id=None):
        """
        Prepare time series data from transactions
        """
        # Convert to dataframe if it's a queryset
        if not isinstance(transactions, pd.DataFrame):
            data = []
            for t in transactions:
                data.append({
                    'date': t.tanggal_transaksi,
                    'product_id': t.stok_id.id if t.stok_id else None,
                    'quantity': t.qty
                })
            df = pd.DataFrame(data)
        else:
            df = transactions
            
        # Filter by product if specified
        if product_id:
            df = df[df['product_id'] == product_id]
            
        # Ensure we have a date column
        if 'date' not in df.columns:
            raise ValueError("Data must contain a 'date' column")
            
        # Resample to daily and fill missing dates
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Aggregate quantity by date (in case multiple transactions per day)
        daily_sales = df.resample('D')['quantity'].sum().fillna(0)
        
        # Store the prepared data
        self.history_data = daily_sales
        
        return daily_sales
    
    def check_stationarity(self, timeseries):
        """
        Check if the time series is stationary using ADF test
        """
        result = adfuller(timeseries.dropna())
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value}')
            
        # Return True if stationary (p-value < 0.05)
        return result[1] < 0.05
    
    def find_optimal_parameters(self):
        """
        Find optimal parameters for the SARIMA model
        """
        # This is a simplified approach - for production, consider grid search
        # with AIC/BIC criteria or auto_arima from pmdarima
        
        data = self.history_data
        
        # Check stationarity
        is_stationary = self.check_stationarity(data)
        
        if not is_stationary:
            print("Series is not stationary, differencing may be needed")
            # Apply differencing
            data_diff = data.diff().dropna()
            is_diff_stationary = self.check_stationarity(data_diff)
            if is_diff_stationary:
                self.difference_order = 1
            else:
                self.difference_order = 2  # Try second order differencing
        
        # Plot ACF and PACF to help determine AR and MA terms
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plot_acf(data.dropna(), ax=plt.gca(), lags=40)
        plt.subplot(212)
        plot_pacf(data.dropna(), ax=plt.gca(), lags=40)
        plt.savefig('acf_pacf_plot.png')
        
        # Based on analysis, we recommend starting parameters
        # In real applications, these should be tuned based on ACF/PACF plots
        # Default parameters
        return {
            'p': 1,  # AR term
            'd': self.difference_order,
            'q': 1,  # MA term
            'P': 1,  # Seasonal AR
            'D': self.seasonal_difference,
            'Q': 1,  # Seasonal MA
            's': self.seasonal_period  # Seasonality period
        }
    
    def train(self, transactions=None, product_id=None, params=None):
        """
        Train SARIMA model with given parameters
        """
        if transactions is not None:
            # Prepare data if new transactions provided
            data = self.prepare_data(transactions, product_id)
        elif self.history_data is not None:
            # Use existing prepared data
            data = self.history_data
        else:
            raise ValueError("No data available for training")
        
        # If no parameters provided, find optimal ones
        if params is None:
            params = self.find_optimal_parameters()
        
        # Extract parameters
        p, d, q = params.get('p', 1), params.get('d', 1), params.get('q', 1)
        P, D, Q, s = params.get('P', 1), params.get('D', 1), params.get('Q', 1), params.get('s', 7)
        
        print(f"Training SARIMA({p},{d},{q})({P},{D},{Q},{s}) model...")
        
        # Train the model
        self.model = SARIMAX(
            data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        print("Model summary:")
        print(self.fitted_model.summary())
        
        return self.fitted_model
    
    def validate(self, test_size=30):
        """
        Validate model performance using historical data
        """
        if self.history_data is None or self.fitted_model is None:
            raise ValueError("Model must be trained first")
        
        # Split data into train and test
        train_data = self.history_data[:-test_size]
        test_data = self.history_data[-test_size:]
        
        # Retrain on training data
        p, d, q = self.fitted_model.specification['order']
        P, D, Q, s = self.fitted_model.specification['seasonal_order']
        
        model_test = SARIMAX(
            train_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results_test = model_test.fit(disp=False)
        
        # Make predictions
        predictions = results_test.get_forecast(steps=test_size)
        predicted_values = predictions.predicted_mean
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, predicted_values))
        mape = np.mean(np.abs((test_data - predicted_values) / test_data)) * 100
        
        print(f"Validation RMSE: {rmse}")
        print(f"Validation MAPE: {mape}%")
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data, label='Actual')
        plt.plot(test_data.index, predicted_values, color='red', label='Predicted')
        plt.title('Model Validation: Predicted vs Actual')
        plt.legend()
        plt.savefig('validation_plot.png')
        
        return {
            'rmse': rmse,
            'mape': mape,
            'actual': test_data,
            'predicted': predicted_values
        }
    
    def predict(self, steps=30):
        """
        Make predictions for future periods
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained first")
            
        # Make forecast
        forecast = self.fitted_model.get_forecast(steps=steps)
        forecast_index = pd.date_range(
            start=self.history_data.index[-1] + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Get predicted values and confidence intervals
        predicted_mean = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        
        # Set proper datetime index
        predicted_mean.index = forecast_index
        confidence_intervals.index = forecast_index
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(self.history_data.index[-90:], self.history_data[-90:], label='Historical')
        plt.plot(predicted_mean.index, predicted_mean, color='red', label='Forecast')
        plt.fill_between(
            confidence_intervals.index,
            confidence_intervals.iloc[:, 0],
            confidence_intervals.iloc[:, 1], 
            color='pink', alpha=0.3
        )
        plt.title('Stock Forecast')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.savefig('forecast_plot.png')
        
        return {
            'forecast': predicted_mean,
            'confidence_intervals': confidence_intervals
        }
    
    def calculate_reorder_point(self, lead_time_days, service_level=0.95, forecast=None):
        """
        Calculate reorder point based on forecast, lead time and service level
        """
        if forecast is None and self.fitted_model is None:
            raise ValueError("Model must be trained first or forecast must be provided")
            
        if forecast is None:
            # Generate forecast for lead time period
            forecast_result = self.predict(steps=lead_time_days)
            forecast = forecast_result['forecast']
        
        # Calculate expected demand during lead time
        expected_demand = forecast[:lead_time_days].sum()
        
        # Calculate standard deviation of forecast errors
        if len(self.history_data) > 30:  # Need enough history
            # Use last 30 days to calculate forecast error
            actual = self.history_data[-30:]
            pred = self.fitted_model.get_prediction(
                start=actual.index[0],
                end=actual.index[-1]
            )
            forecast_errors = actual - pred.predicted_mean
            std_dev = forecast_errors.std()
            
            # Calculate safety stock based on service level
            from scipy import stats
            z_score = stats.norm.ppf(service_level)
            safety_stock = z_score * std_dev * np.sqrt(lead_time_days)
            
            # Reorder point = expected demand during lead time + safety stock
            reorder_point = expected_demand + safety_stock
        else:
            # Simplified calculation if not enough history
            reorder_point = expected_demand * 1.5  # 50% buffer
            
        return {
            'expected_demand': expected_demand,
            'reorder_point': reorder_point
        }