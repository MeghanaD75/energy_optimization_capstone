import pandas as pd

def monitor_energy_usage(file):
    df = pd.read_csv(file)
    return df.tail(5)
