import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class WAQIDataFetcher:
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.waqi.info"
    
    def fetch_station_data(self, station_id: str) -> Optional[Dict]:
        url = f"{self.base_url}/feed/{station_id}/"
        params = {"token": self.api_token}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                return data.get("data")
            else:
                print(f"API Error: {data.get('data')}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def parse_station_data(self, raw_data: Dict) -> Dict:
        if not raw_data:
            return {}
        
        iaqi = raw_data.get('iaqi', {})
        city_info = raw_data.get('city', {})
        geo = city_info.get('geo', [None, None])
        time_info = raw_data.get('time', {})
        
        parsed = {
            'timestamp': time_info.get('s'),
            'timestamp_unix': time_info.get('v'),
            'aqi': raw_data.get('aqi'),
            'station_name': city_info.get('name'),
            'station_url': city_info.get('url'),
            'latitude': geo[0] if len(geo) > 0 else None,
            'longitude': geo[1] if len(geo) > 1 else None,
            'pm25': iaqi.get('pm25', {}).get('v'),
            'pm10': iaqi.get('pm10', {}).get('v'),
            'o3': iaqi.get('o3', {}).get('v'),
            'no2': iaqi.get('no2', {}).get('v'),
            'so2': iaqi.get('so2', {}).get('v'),
            'co': iaqi.get('co', {}).get('v'),
            'temperature': iaqi.get('t', {}).get('v'),
            'pressure': iaqi.get('p', {}).get('v'),
            'humidity': iaqi.get('h', {}).get('v'),
            'wind_speed': iaqi.get('w', {}).get('v'),
            'dew_point': iaqi.get('dew', {}).get('v'),
        }
        
        return parsed
    
    def search_city(self, city_name: str):
        url = f"{self.base_url}/search/"
        params = {"token": self.api_token, "keyword": city_name}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if data.get("status") == "ok":
                return data.get("data", [])
        except:
            pass
        return None


class SmartNullHandler:
    
    def __init__(self, strategy: str = 'smart'):
        self.strategy = strategy
        self.imputation_values = {}
    
    def analyze_nulls(self, df: pd.DataFrame) -> Dict:
        null_report = {
            'total_nulls': df.isnull().sum().sum(),
            'null_by_column': df.isnull().sum().to_dict(),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'complete_columns': df.columns[~df.isnull().any()].tolist()
        }
        return null_report
    
    def handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if self.strategy == 'smart':
            df = self._smart_strategy(df)
        elif self.strategy == 'fill':
            df = self._fill_strategy(df)
        elif self.strategy == 'drop':
            df = self._drop_strategy(df)
        elif self.strategy == 'interpolate':
            df = self._interpolate_strategy(df)
        
        return df
    
    def _smart_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if 'aqi' in df.columns and df['aqi'].isna().any():
            print("Warning: AQI is missing. This record may be invalid.")
            df['aqi'] = df['aqi'].fillna(-1)
            df['data_quality'] = 'invalid'
        else:
            df['data_quality'] = 'valid'
        
        pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for pollutant in pollutants:
            if pollutant in df.columns:
                df[f'{pollutant}_imputed'] = df[pollutant].isna().astype(int)
                
                if df[pollutant].notna().any():
                    df[pollutant] = df[pollutant].fillna(df[pollutant].median())
                else:
                    df[pollutant] = df[pollutant].fillna(0)
        
        weather_defaults = {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.25,
            'wind_speed': 2.0,
            'dew_point': 15.0
        }
        
        for weather_param, default_val in weather_defaults.items():
            if weather_param in df.columns:
                df[f'{weather_param}_imputed'] = df[weather_param].isna().astype(int)
                df[weather_param] = df[weather_param].fillna(default_val)
        
        imputed_cols = [col for col in df.columns if col.endswith('_imputed')]
        df['total_imputed_features'] = df[imputed_cols].sum(axis=1) if imputed_cols else 0
        
        return df
    
    def _fill_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        return df
    
    def _drop_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)
        df = df.dropna()
        print(f"Dropped {original_len - len(df)} rows due to nulls")
        return df
    
    def _interpolate_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        return df


class AQIExploratoryAnalysis:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
    
    def run_complete_eda(self, save_plots: bool = False):
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS - AIR QUALITY INDEX")
        print("="*70 + "\n")
        
        self._basic_info()
        self._missing_values_analysis()
        self._data_quality_check()
        self._aqi_analysis()
        self._pollutant_analysis()
        self._weather_analysis()
        self._correlation_analysis()
        self._web_app_readiness_check()
        
        if save_plots:
            self._create_visualizations()
        
        return self.results
    
    def _basic_info(self):
        print("BASIC INFORMATION")
        print("-" * 70)
        print(f"Dataset Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if 'station_name' in self.df.columns:
            print(f"Station: {self.df['station_name'].iloc[0]}")
        print()
    
    def _missing_values_analysis(self):
        print("MISSING VALUES ANALYSIS")
        print("-" * 70)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct,
            'Dtype': self.df.dtypes
        }).sort_values('Missing_Count', ascending=False)
        
        if missing_df['Missing_Count'].sum() > 0:
            print(missing_df[missing_df['Missing_Count'] > 0])
            print(f"\nTotal Missing Values: {missing_df['Missing_Count'].sum()}")
        else:
            print("No missing values found")
        
        self.results['missing_values'] = missing_df
        print()
    
    def _data_quality_check(self):
        print("DATA QUALITY CHECK")
        print("-" * 70)
        
        if 'data_quality' in self.df.columns:
            quality_counts = self.df['data_quality'].value_counts()
            print("Data Quality Status:")
            for status, count in quality_counts.items():
                print(f"  {status.upper()}: {count} records")
        
        if 'total_imputed_features' in self.df.columns:
            avg_imputed = self.df['total_imputed_features'].mean()
            print(f"\nAverage Imputed Features per Record: {avg_imputed:.2f}")
            
            if avg_imputed > 5:
                print("Warning: High imputation rate - API may have sensor issues")
        print()
    
    def _aqi_analysis(self):
        if 'aqi' not in self.df.columns or self.df['aqi'].isna().all():
            print("No AQI data available\n")
            return
        
        print("AQI ANALYSIS")
        print("-" * 70)
        
        aqi_val = self.df['aqi'].iloc[0] if len(self.df) > 0 else None
        
        if aqi_val and aqi_val != -1:
            print(f"Current AQI: {aqi_val}")
            
            category, health_msg = self._categorize_aqi(aqi_val)
            print(f"Category: {category}")
            print(f"Health Implication: {health_msg}")
            
            self.results['aqi_category'] = category
            self.results['current_aqi'] = aqi_val
            self.results['health_message'] = health_msg
        else:
            print("Invalid AQI value")
        print()
    
    def _categorize_aqi(self, aqi: float) -> Tuple[str, str]:
        if aqi <= 50:
            return ("Good", "Air quality is satisfactory")
        elif aqi <= 100:
            return ("Moderate", "Acceptable for most people")
        elif aqi <= 150:
            return ("Unhealthy for Sensitive Groups", 
                   "Sensitive groups may experience effects")
        elif aqi <= 200:
            return ("Unhealthy", "Everyone may experience effects")
        elif aqi <= 300:
            return ("Very Unhealthy", "Health alert for everyone")
        else:
            return ("Hazardous", "Emergency conditions")
    
    def _pollutant_analysis(self):
        print("POLLUTANT ANALYSIS")
        print("-" * 70)
        
        pollutants = {
            'pm25': ('PM2.5', 'Fine Particulate Matter'),
            'pm10': ('PM10', 'Coarse Particulate Matter'),
            'o3': ('O3', 'Ozone'),
            'no2': ('NO2', 'Nitrogen Dioxide'),
            'so2': ('SO2', 'Sulfur Dioxide'),
            'co': ('CO', 'Carbon Monoxide')
        }
        
        available = []
        for col, (symbol, name) in pollutants.items():
            if col in self.df.columns and self.df[col].notna().any():
                val = self.df[col].iloc[0]
                imputed = self.df[f'{col}_imputed'].iloc[0] if f'{col}_imputed' in self.df.columns else 0
                status = " (imputed)" if imputed else ""
                print(f"{symbol} - {name}: {val}{status}")
                available.append(col)
        
        if not available:
            print("No pollutant data available")
        
        self.results['available_pollutants'] = available
        print()
    
    def _weather_analysis(self):
        print("WEATHER ANALYSIS")
        print("-" * 70)
        
        weather_params = {
            'temperature': ('Temperature', 'C'),
            'humidity': ('Humidity', '%'),
            'pressure': ('Pressure', 'hPa'),
            'wind_speed': ('Wind Speed', 'm/s'),
            'dew_point': ('Dew Point', 'C')
        }
        
        for col, (name, unit) in weather_params.items():
            if col in self.df.columns and self.df[col].notna().any():
                val = self.df[col].iloc[0]
                imputed = self.df[f'{col}_imputed'].iloc[0] if f'{col}_imputed' in self.df.columns else 0
                status = " (imputed)" if imputed else ""
                print(f"{name}: {val} {unit}{status}")
        print()
    
    def _correlation_analysis(self):
        print("CORRELATION ANALYSIS")
        print("-" * 70)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols 
                       if not col.endswith('_imputed') 
                       and col not in ['latitude', 'longitude', 'timestamp_unix']]
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            
            if 'aqi' in corr_matrix.columns:
                aqi_corr = corr_matrix['aqi'].sort_values(ascending=False)
                print("Correlation with AQI:")
                for feature, corr_val in aqi_corr.items():
                    if feature != 'aqi' and abs(corr_val) > 0.3:
                        print(f"  {feature}: {corr_val:.3f}")
            
            self.results['correlation_matrix'] = corr_matrix
        print()
    
    def _web_app_readiness_check(self):
        print("WEB APP READINESS CHECK")
        print("-" * 70)
        
        checks = {
            'Has AQI': 'aqi' in self.df.columns and self.df['aqi'].notna().all(),
            'Has Timestamp': 'timestamp' in self.df.columns,
            'Has Location': 'latitude' in self.df.columns and 'longitude' in self.df.columns,
            'Has Station Name': 'station_name' in self.df.columns,
            'Data Quality OK': 'data_quality' not in self.df.columns or 
                              (self.df['data_quality'] == 'valid').all()
        }
        
        all_passed = all(checks.values())
        
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {check}")
        
        print()
        if all_passed:
            print("Data is READY for web app deployment")
        else:
            print("Some checks failed - review data quality")
        
        self.results['web_app_ready'] = all_passed
        print()
    
    def _create_visualizations(self):
        try:
            plt.figure(figsize=(10, 6))
            plt.savefig('eda_plots.png')
            print("Visualizations saved to: eda_plots.png")
        except:
            print("Skipping visualizations (no display available)")


class FeatureEngineer:
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            df['time_of_day'] = pd.cut(df['hour'], 
                                       bins=[0, 6, 12, 18, 24],
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)
            
            df['time_of_day_numeric'] = pd.cut(df['hour'], 
                                               bins=[0, 6, 12, 18, 24],
                                               labels=[0, 1, 2, 3],
                                               include_lowest=True).astype(int)
        
        if 'aqi' in df.columns:
            def categorize_aqi(aqi):
                if pd.isna(aqi) or aqi == -1: return 0
                if aqi <= 50: return 1
                if aqi <= 100: return 2
                if aqi <= 150: return 3
                if aqi <= 200: return 4
                if aqi <= 300: return 5
                return 6
            
            df['aqi_category_numeric'] = df['aqi'].apply(categorize_aqi)
        
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        if 'wind_speed' in df.columns and 'pm25' in df.columns:
            df['wind_pm25_interaction'] = df['wind_speed'] * df['pm25']
        
        if 'pressure' in df.columns and 'temperature' in df.columns:
            df['pressure_temp_ratio'] = df['pressure'] / (df['temperature'] + 273.15)
        
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['discomfort_index'] = (df['temperature'] - 
                                     (0.55 - 0.0055 * df['humidity']) * 
                                     (df['temperature'] - 14.5))
        
        return df


class HopsworksFeatureStore:
    
    def __init__(self, api_key: str, project_name: str):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.fs = None
    
    def connect(self):
        print("\nConnecting to Hopsworks...")
        try:
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            print("Successfully connected to Hopsworks")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def create_feature_group(self, df: pd.DataFrame, 
                        feature_group_name: str = "aqi_features",
                        version: int = 1,
                        description: str = "Air Quality Index features"):
        
        if self.fs is None:
            print("Not connected to Hopsworks. Call connect() first.")
            return None
        
        print(f"\nCreating feature group: {feature_group_name}")
        
        try:
            feature_group = self.fs.get_or_create_feature_group(
                name=feature_group_name,
                version=version,
                description=description,
                primary_key=['timestamp'],
                event_time='datetime' if 'datetime' in df.columns else None,
                online_enabled=False
            )
            
            print("Inserting data into feature store...")
            feature_group.insert(df, write_options={"wait_for_job": True})
            
            print(f"Successfully created feature group: {feature_group_name} v{version}")
            print(f"Inserted {len(df)} records with {len(df.columns)} features")
            
            return feature_group
            
        except Exception as e:
            print(f"Error creating feature group: {e}")
            return None


def main():
    WAQI_API_TOKEN = "088661c637816f9f1463ca3e44d37da6d739d021"
    STATION_ID = "A401143"
    HOPSWORKS_API_KEY = "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5"
    HOPSWORKS_PROJECT = "AQI_Project_10"
    NULL_HANDLING_STRATEGY = "smart"

    print("--------------------------------------------------")
    print("Fetching data...")
    
    fetcher = WAQIDataFetcher(WAQI_API_TOKEN)
    raw_data = fetcher.fetch_station_data(STATION_ID)
    
    if not raw_data:
        print("Failed to fetch data")
        return
        
    parsed_data = fetcher.parse_station_data(raw_data)
    df_raw = pd.DataFrame([parsed_data])
    
    print(f"Station: {parsed_data.get('station_name')}")
    print("--------------------------------------------------")
    
    null_handler = SmartNullHandler(strategy=NULL_HANDLING_STRATEGY)
    df_cleaned = null_handler.handle_nulls(df_raw)
    
    df_features = FeatureEngineer.create_features(df_cleaned)
    
    print("Connecting to Hopsworks...")
    hops = HopsworksFeatureStore(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)
    
    if hops.connect():
        feature_group = hops.create_feature_group(
            df=df_features,
            feature_group_name="aqi_features",
            version=1,
            description="Air Quality Index features"
        )
        if feature_group:
            print("Data uploaded successfully")
            print("--------------------------------------------------")
    else:
        print("Failed to connect to Hopsworks")
        df_features.to_csv("aqi_features_today.csv", index=False)
        print("Saved data locally to: aqi_features_today.csv")
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()