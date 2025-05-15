def generate_report(df):
    return f"Total Usage: {df['energy_kwh'].sum()} kWh\nAnomalies: {len(df[df['anomaly']])}"
