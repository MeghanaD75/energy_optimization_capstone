def detect_anomalies(df):
    df['anomaly'] = df['energy_kwh'] > df['energy_kwh'].mean() * 1.5
    return df[df['anomaly']]
