import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import warnings, itertools, logging
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.model = None
        self.history_data = None
        self.difference_order = 1  # Default differencing order
        self.seasonal_difference = 1  # Default seasonal differencing
        self.seasonal_period = 7  # Default seasonality (weekly)
        
    def prepare_data(self, transactions, product_id=None, type_filter=2, handle_outliers=False):
        """
        Prepare time series data from transactions
        
        Args:
            transactions: DataFrame or queryset of transactions
            product_id: Optional filter for specific product
            type_filter: Transaction type to filter (2 = sales)
            handle_outliers: Whether to detect and handle outliers
        
        Returns:
            pandas.Series: Daily sales data
        """
        # Convert to dataframe if it's a queryset
        if not isinstance(transactions, pd.DataFrame):
            data = []
            for t in transactions:
                data.append({
                    'date': t.tanggal_transaksi,
                    'product_id': t.stok_id.id if t.stok_id else None,
                    'quantity': t.qty,
                    'type': t.tipe
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
        
        if 'type' in df.columns and type_filter:
            df = df[df['type'] == type_filter]
            
        # Resample to daily and fill missing dates
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Aggregate quantity by date (in case multiple transactions per day)
        daily_sales = df.resample('D')['quantity'].sum().fillna(0)
        
        # Handle outliers if requested
        if handle_outliers and len(daily_sales) >= 30:
            daily_sales = self._handle_outliers(daily_sales)
        
        # Store the prepared data
        self.history_data = daily_sales
        
        return daily_sales
    
    def _handle_outliers(self, timeseries, threshold=3):
        """
        Detect and handle outliers in the time series
        
        Args:
            timeseries: The time series data
            threshold: Z-score threshold for outlier detection
            
        Returns:
            pandas.Series: Time series with outliers handled
        """
        # Only consider non-zero values for outlier detection
        non_zero_mask = timeseries > 0
        non_zero_values = timeseries[non_zero_mask]
        
        if len(non_zero_values) < 5:  # Need enough data points
            return timeseries
            
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(non_zero_values))
        outliers_mask = z_scores > threshold
        
        if not outliers_mask.any():
            return timeseries
            
        # Create a copy to avoid modifying the original
        clean_series = timeseries.copy()
        
        # Replace outliers with median of surrounding values
        outlier_indices = non_zero_values.index[outliers_mask]
        for idx in outlier_indices:
            # Get values from 3 days before and after
            window_size = 3
            window_start = timeseries.index.get_loc(idx) - window_size
            window_end = timeseries.index.get_loc(idx) + window_size + 1
            
            window_start = max(0, window_start)
            window_end = min(len(timeseries), window_end)
            
            surrounding_values = timeseries.iloc[window_start:window_end]
            # Only use non-zero values for calculating median
            non_zero_window = surrounding_values[surrounding_values > 0]
            
            if len(non_zero_window) > 0:
                median_value = non_zero_window.median()
                clean_series.loc[idx] = median_value
                logger.info(f"Replaced outlier {timeseries.loc[idx]} with {median_value} at {idx}")
        
        return clean_series
    
    def check_stationarity(self, timeseries):
        """
        Check if the time series is stationary using ADF test
        Returns True if stationary
        """
        # Handle case when all values are 0
        if all(v == 0 for v in timeseries):
            return True
            
        # Need at least some non-zero values
        non_zero_values = timeseries[timeseries > 0]
        if len(non_zero_values) < 10:
            return False
            
        try:
            result = adfuller(timeseries.dropna())
            logger.info(f'ADF Statistic: {result[0]}')
            logger.info(f'p-value: {result[1]}')
            
            # Return True if stationary (p-value < 0.05)
            return result[1] < 0.05
        except Exception as e:
            logger.warning(f"Error in stationarity check: {str(e)}")
            return False
    
    def find_optimal_parameters(self, max_p=2, max_q=2, max_d=1, seasonal_period=7):
        """
        Find optimal SARIMA parameters using grid search
        
        Args:
            max_p: Maximum AR order
            max_q: Maximum MA order
            max_d: Maximum differencing
            seasonal_period: Seasonal period (7=weekly, 30=monthly)
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Searching optimal SARIMA parameters with grid search...")
        
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None

        # Parameter range
        p = range(0, max_p + 1)
        d = range(0, max_d + 1)
        q = range(0, max_q + 1)
        
        # Seasonal components - limit search to reduce computation time
        D = range(0, 2)
        P = range(0, 2)
        Q = range(0, 2)
        s = seasonal_period
        
        # Handle very short time series
        if len(self.history_data) < 2 * seasonal_period:
            logger.warning("Time series too short for seasonal model, using non-seasonal model")
            s = 0
            
        # First check stationarity to determine differencing
        is_stationary = self.check_stationarity(self.history_data)
        if is_stationary:
            logger.info("Series is stationary, limiting differencing")
            d = range(0, 1)  # No or minimal differencing needed
        
        # Try to use non-seasonal model first if data is sparse
        if (self.history_data == 0).mean() > 0.5:
            logger.info("Sparse data detected, trying non-seasonal model first")
            models_to_try = [
                # Non-seasonal models
                [(p_val, d_val, q_val) for p_val in p for d_val in d for q_val in q],
                # Seasonal models
                [(p_val, d_val, q_val, P_val, D_val, Q_val, s) 
                 for p_val in p for d_val in d for q_val in q 
                 for P_val in P for D_val in D for Q_val in Q]
            ]
        else:
            models_to_try = [
                # Try seasonal models first
                [(p_val, d_val, q_val, P_val, D_val, Q_val, s) 
                 for p_val in p for d_val in d for q_val in q 
                 for P_val in P for D_val in D for Q_val in Q],
                # Non-seasonal models as fallback
                [(p_val, d_val, q_val) for p_val in p for d_val in d for q_val in q]
            ]
            
        # Limit number of models to try if the data is very long
        if len(self.history_data) > 365:
            logger.info("Long time series detected, limiting parameter search")
            for i in range(len(models_to_try)):
                models_to_try[i] = models_to_try[i][:20]

        for models in models_to_try:
            for model_params in models:
                try:
                    if len(model_params) == 3:  # Non-seasonal model
                        model = SARIMAX(
                            self.history_data,
                            order=model_params,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        seasonal_params = (0, 0, 0, 0)
                    else:  # Seasonal model
                        order = model_params[:3]
                        seasonal_order = model_params[3:]
                        model = SARIMAX(
                            self.history_data,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        seasonal_params = seasonal_order
                        
                    result = model.fit(disp=False, maxiter=50)
                    
                    # Check if this model is better
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = model_params[:3]
                        if len(model_params) > 3:
                            best_seasonal_order = model_params[3:]
                        else:
                            best_seasonal_order = (0, 0, 0, 0)
                            
                    # Early stopping if we found a good model
                    if best_aic < float('inf') and len(models) > 10:
                        # If we've checked at least 10 models and found one that works
                        if model_params == models[9]:
                            break
                except Exception as e:
                    logger.debug(f"Error fitting model with params {model_params}: {str(e)}")
                    continue
                    
            # If we found a good model in the first group, don't try the second
            if best_aic < float('inf'):
                break

        if best_order is None:
            logger.warning("Could not find optimal parameters, using defaults")
            best_order = (1, 1, 1)
            best_seasonal_order = (0, 0, 0, 0)
            
        logger.info(f"Best AIC: {best_aic}")
        logger.info(f"Best order: {best_order}")
        if len(best_seasonal_order) == 4:
            logger.info(f"Best seasonal order: ({best_seasonal_order[0]},{best_seasonal_order[1]},{best_seasonal_order[2]},{best_seasonal_order[3]})")
        
        # Return as dictionary
        if len(best_seasonal_order) == 4:
            return {
                'p': best_order[0], 'd': best_order[1], 'q': best_order[2],
                'P': best_seasonal_order[0], 'D': best_seasonal_order[1], 
                'Q': best_seasonal_order[2], 's': best_seasonal_order[3]
            }
        else:
            return {
                'p': best_order[0], 'd': best_order[1], 'q': best_order[2],
                'P': 0, 'D': 0, 'Q': 0, 's': 0  # No seasonality
            }
    
    def train(self, transactions=None, product_id=None, params=None):
        """
        Train SARIMA model with given parameters
        
        Args:
            transactions: Optional new transactions data
            product_id: Optional product ID filter
            params: Model parameters dictionary
            
        Returns:
            statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper: Fitted model
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
        
        logger.info(f"Training SARIMA({p},{d},{q})({P},{D},{Q},{s}) model...")
        
        # Train the model
        try:
            self.model = SARIMAX(
                data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False)
            logger.info("Model training completed successfully")
            
            # Validate model
            residuals = self.fitted_model.resid
            
            # Check if residuals have high autocorrelation (indicating poor fit)
            if len(residuals) > 10:
                residual_acf = acf(residuals, nlags=10)
                if abs(residual_acf[1:]).max() > 0.3:
                    logger.warning("High autocorrelation in residuals detected, model may not be optimal")
            
            return self.fitted_model
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            # Fallback to simpler model
            logger.info("Trying simpler model...")
            try:
                self.model = SARIMAX(
                    data,
                    order=(1, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_model = self.model.fit(disp=False)
                return self.fitted_model
            except Exception as e2:
                logger.error(f"Error training fallback model: {str(e2)}")
                raise ValueError(f"Could not train prediction model: {str(e2)}")
    
    def validate(self, test_size=30):
        """
        Validate model performance using cross-validation
        
        Args:
            test_size: Number of days to use for testing
            
        Returns:
            dict: Validation metrics
        """
        if self.history_data is None or self.fitted_model is None:
            raise ValueError("Model must be trained first")
            
        # Ensure test size is reasonable
        test_size = min(test_size, len(self.history_data) // 3)
        if test_size < 7:
            logger.warning("Test size too small for validation, using 7 days")
            test_size = 7
        
        # Split data into train and test
        train_data = self.history_data[:-test_size]
        test_data = self.history_data[-test_size:]
        
        if len(train_data) < 2 * 7:  # Need at least 2 weeks for training
            logger.warning("Not enough data for validation")
            return {
                'rmse': float('nan'),
                'mape': float('nan'),
                'actual': test_data,
                'predicted': pd.Series(index=test_data.index, data=np.zeros_like(test_data))
            }
        
        # Retrain on training data
        p, d, q = self.fitted_model.specification['order']
        P, D, Q, s = self.fitted_model.specification['seasonal_order']
        
        try:
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
            
            # Ensure predicted values have the same index as test data
            predicted_values.index = test_data.index
            
            # Calculate metrics
            # Handle cases with zeros in test data for MAPE
            non_zero_mask = test_data != 0
            if non_zero_mask.any():
                mape = mean_absolute_percentage_error(
                    test_data[non_zero_mask], 
                    predicted_values[non_zero_mask]
                ) * 100
            else:
                mape = float('nan')
                
            rmse = np.sqrt(mean_squared_error(test_data, predicted_values))
            
            logger.info(f"Validation RMSE: {rmse}")
            logger.info(f"Validation MAPE: {mape}%")
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.plot(test_data.index, test_data, label='Actual')
            plt.plot(test_data.index, predicted_values, color='red', label='Predicted')
            plt.title('Model Validation: Predicted vs Actual')
            plt.legend()
            plt.savefig('validation_plot.png')
            plt.close()
            
            return {
                'rmse': rmse,
                'mape': mape,
                'actual': test_data,
                'predicted': predicted_values
            }
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return {
                'rmse': float('nan'),
                'mape': float('nan'),
                'error': str(e)
            }
    
    def predict(self, steps=30):
        """
        Make predictions for future periods
        
        Args:
            steps: Number of days to forecast
            
        Returns:
            dict: Forecast results with mean values and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained first")
            
        # Make forecast
        try:
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
            
            # Ensure no negative values in forecast (can't have negative sales)
            predicted_mean = predicted_mean.clip(lower=0)
            confidence_intervals.iloc[:, 0] = confidence_intervals.iloc[:, 0].clip(lower=0)
            
            # Plot forecast
            plt.figure(figsize=(12, 6))
            history_points = min(90, len(self.history_data))
            plt.plot(self.history_data[-history_points:].index, 
                     self.history_data[-history_points:], label='Historical')
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
            plt.close()
            
            return {
                'forecast': predicted_mean,
                'confidence_intervals': confidence_intervals
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fallback to naive forecast (moving average)
            logger.info("Using fallback moving average forecast")
            
            last_n = min(30, len(self.history_data))
            moving_avg = self.history_data[-last_n:].mean()
            
            # Create forecast series
            forecast_index = pd.date_range(
                start=self.history_data.index[-1] + timedelta(days=1),
                periods=steps,
                freq='D'
            )
            forecast_values = pd.Series(
                index=forecast_index,
                data=[moving_avg] * steps
            )
            
            # Create confidence intervals (just +/- 50% of moving average)
            lower = forecast_values * 0.5
            upper = forecast_values * 1.5
            conf_int = pd.DataFrame(
                {'lower': lower, 'upper': upper},
                index=forecast_index
            )
            
            return {
                'forecast': forecast_values,
                'confidence_intervals': conf_int
            }
    
    def calculate_reorder_point(self, lead_time_days, service_level=0.95, forecast=None):
        """
        Calculate reorder point based on forecast, lead time and service level
        
        Args:
            lead_time_days: Number of days to receive new stock
            service_level: Desired service level (default 0.95 or 95%)
            forecast: Optional forecast data
            
        Returns:
            dict: Reorder point calculation results
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
            try:
                # Use last 30 days to calculate forecast error
                actual = self.history_data[-30:]
                pred = self.fitted_model.get_prediction(
                    start=actual.index[0],
                    end=actual.index[-1]
                )
                forecast_errors = actual - pred.predicted_mean
                std_dev = forecast_errors.std()
                
                # If std_dev is too small, use a minimum value based on data variability
                min_std = actual.std() * 0.2
                std_dev = max(std_dev, min_std)
                
                # Calculate safety stock based on service level
                z_score = stats.norm.ppf(service_level)
                safety_stock = z_score * std_dev * np.sqrt(lead_time_days)
                
                # Reorder point = expected demand during lead time + safety stock
                reorder_point = expected_demand + safety_stock
            except Exception as e:
                logger.error(f"Error calculating reorder point: {str(e)}")
                # Fallback calculation
                reorder_point = expected_demand * 1.5  # 50% buffer
        else:
            # Simplified calculation if not enough history
            reorder_point = expected_demand * 1.5  # 50% buffer
            
        return {
            'expected_demand': expected_demand,
            'reorder_point': reorder_point
        }

    def get_best_restock_day(self, forecast_days=30, lead_time=7):
        """
        Find the optimal day to restock based on forecast pattern
        
        Args:
            forecast_days: Number of days to look ahead
            lead_time: Lead time for delivery
            
        Returns:
            int: Recommended days from now to place order
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained first")
            
        # Get forecast
        forecast_result = self.predict(steps=forecast_days)
        forecast = forecast_result['forecast']
        
        # Find days with lowest demand (best for receiving stock)
        daily_avg = forecast.groupby(forecast.index.dayofweek).mean()
        best_day_of_week = daily_avg.idxmin()
        
        # Find the next occurrence of the best day
        today_dow = datetime.now().weekday()
        days_until_best = (best_day_of_week - today_dow) % 7
        
        # Adjust for lead time
        reorder_day = (days_until_best - lead_time) % 7
        if reorder_day == 0:
            reorder_day = 7  # Order today if calculation gives 0
            
        return reorder_day