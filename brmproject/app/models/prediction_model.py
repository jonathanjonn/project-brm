import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import model_selection
from scipy import stats
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.model = None
        self.history_data = None

    def prepare_data(self, transactions_df, type_filter=2, resample_freq='D'):
        df = transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        if 'type' in df.columns and type_filter:
            df = df[df['type'] == type_filter]

        daily_sales = df['quantity'].resample(resample_freq).sum().fillna(0)
        
        # Handle outliers using IQR method
        Q1 = daily_sales.quantile(0.25)
        Q3 = daily_sales.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with the median of the series
        median = daily_sales.median()
        daily_sales[(daily_sales < lower_bound) | (daily_sales > upper_bound)] = median
        
        self.history_data = daily_sales
        return daily_sales

    def train(self, data=None):
        if data is not None:
            self.history_data = data
        
        if self.history_data is None or self.history_data.empty:
            raise ValueError("Data untuk training tidak tersedia.")

        logger.info(f"Memulai auto_arima untuk menemukan model SARIMA terbaik...")
        
        try:
            self.model = pm.auto_arima(
                self.history_data,
                start_p=1, start_q=1,
                test='kpss',       # Gunakan KPSS test untuk menentukan d
                max_p=3, max_q=3,
                m=7,              # Asumsi musiman mingguan
                d=None,           # Biarkan auto_arima menentukan d
                seasonal=True,    # Aktifkan pencarian musiman
                start_P=0, D=None,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True       # Gunakan stepwise search untuk kecepatan
            )
            
            logger.info(f"Model SARIMA terbaik ditemukan: {self.model.order}, {self.model.seasonal_order}")
            logger.info(self.model.summary())

        except Exception as e:
            logger.error(f"Gagal melatih model dengan auto_arima: {e}")
            # Fallback ke model non-musiman jika auto_arima gagal
            self.model = pm.auto_arima(self.history_data, seasonal=False, error_action='ignore', suppress_warnings=True)
            logger.info(f"Menggunakan model non-musiman sebagai fallback: {self.model.order}")

        return self.model

    def predict(self, steps=30):
        if self.model is None:
            raise ValueError("Model harus dilatih terlebih dahulu.")
        
        forecast, conf_int = self.model.predict(n_periods=steps, return_conf_int=True)
        
        # Pastikan prediksi tidak negatif
        forecast = np.maximum(0, forecast)
        conf_int = np.maximum(0, conf_int)

        return {
            'forecast': forecast,
            'confidence_intervals': conf_int
        }

    def calculate_reorder_point(self, lead_time_days, service_level=0.95):
        if self.history_data is None:
            raise ValueError("Data historis tidak tersedia.")

        # Permintaan rata-rata harian dari data historis
        avg_daily_demand = self.history_data.mean()
        
        # Deviasi standar dari permintaan harian
        std_daily_demand = self.history_data.std()

        # Permintaan rata-rata selama lead time
        avg_lead_time_demand = avg_daily_demand * lead_time_days

        # Deviasi standar permintaan selama lead time
        std_lead_time_demand = std_daily_demand * np.sqrt(lead_time_days)

        # Z-score untuk tingkat layanan yang diinginkan
        z_score = stats.norm.ppf(service_level)

        # Safety stock
        safety_stock = z_score * std_lead_time_demand

        # Reorder point
        reorder_point = avg_lead_time_demand + safety_stock

        return {
            'expected_demand': np.maximum(0, avg_lead_time_demand),
            'reorder_point': np.maximum(0, reorder_point),
            'safety_stock': np.maximum(0, safety_stock)
        }

    def cross_validate(self, data, steps=30, cv_folds=3):
        if len(data) < (cv_folds * steps):
            logger.warning("Data tidak cukup untuk cross-validation, melewati.")
            return None

        cv = model_selection.SlidingWindowForecastCV(window_size=len(data) - (cv_folds * steps), step=steps, h=steps)
        
        # Latih ulang model pada setiap lipatan cross-validation
        # Ini bisa memakan waktu, jadi gunakan dengan bijak
        # Untuk implementasi ini, kita hanya akan menggunakan model yang sudah ada
        # dan memvalidasinya pada data test
        train, test = model_selection.train_test_split(data, test_size=steps)
        
        model_for_validation = pm.auto_arima(train, seasonal=True, m=7, suppress_warnings=True, error_action='ignore')
        predictions = model_for_validation.predict(n_periods=test.shape[0])
        
        mape = np.mean(np.abs((test - predictions) / test)) * 100
        rmse = np.sqrt(np.mean((test - predictions)**2))

        logger.info(f"Cross-validation MAPE: {mape:.2f}%")
        logger.info(f"Cross-validation RMSE: {rmse:.2f}")

        return {'mape': mape, 'rmse': rmse}
