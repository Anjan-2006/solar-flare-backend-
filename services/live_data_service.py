import requests
import pandas as pd
import numpy as np

class LiveDataService:
    XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"

    @classmethod
    def fetch_live_data(cls) -> pd.DataFrame:
        try:
            xray_data = requests.get(cls.XRAY_URL).json()
            plasma_data = requests.get(cls.PLASMA_URL).json()
            mag_data = requests.get(cls.MAG_URL).json()

            # Process X-Ray Data
            df_xray = pd.DataFrame(xray_data)
            # FIX: Ensure UTC aware
            df_xray['time_tag'] = pd.to_datetime(df_xray['time_tag'], utc=True)
            df_xray = df_xray.pivot_table(index='time_tag', columns='energy', values='flux').reset_index()
            
            df_xray.rename(columns={
                '0.05-0.4nm': 'xrsa_flux',
                '0.1-0.8nm': 'xrsb_flux'
            }, inplace=True)
            
            if 'xrsa_flux' not in df_xray.columns:
                df_xray['xrsa_flux'] = 1e-9
            if 'xrsb_flux' not in df_xray.columns:
                df_xray['xrsb_flux'] = 1e-8

            df_xray.set_index('time_tag', inplace=True)

            # Process Plasma Data
            df_plasma = pd.DataFrame(plasma_data[1:], columns=plasma_data[0])
            # FIX: Ensure UTC aware
            df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'], utc=True)
            df_plasma.rename(columns={
                'density': 'proton_density',
                'speed': 'flow_speed',
                'temperature': 'proton_temperature'
            }, inplace=True)
            df_plasma.set_index('time_tag', inplace=True)

            # Process Mag Data
            df_mag = pd.DataFrame(mag_data[1:], columns=mag_data[0])
            # FIX: Ensure UTC aware
            df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'], utc=True)
            df_mag.rename(columns={
                'bx_gsm': 'Bx',
                'by_gsm': 'By',
                'bz_gsm': 'Bz',
                'bt': 'IMF_magnitude'
            }, inplace=True)
            df_mag.set_index('time_tag', inplace=True)

            # Merge all on exact time_tag (all now UTC-aware)
            df_merged = df_xray.join(df_plasma, how='outer').join(df_mag, how='outer')
            df_merged.sort_index(inplace=True)
            
            df_merged = df_merged.apply(pd.to_numeric, errors='coerce')
            df_merged.ffill(inplace=True)
            df_merged.bfill(inplace=True)

            required_cols = [
                'xrsa_flux', 'xrsb_flux', 'IMF_magnitude', 'Bx', 'By', 'Bz',
                'flow_speed', 'proton_density', 'proton_temperature'
            ]
            
            for col in required_cols:
                if col not in df_merged.columns:
                    df_merged[col] = 0.0

            df_final = df_merged[required_cols].copy()
            df_final = df_final.tail(200).reset_index(drop=True)

            if df_final.empty:
                raise ValueError("No data available after merging NOAA datasets.")

            return df_final

        except Exception as e:
            raise ValueError(f"Failed to fetch or process NOAA live data: {str(e)}")
