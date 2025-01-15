import json
import pandas as pd
import numpy as np
from prov4ml.utils.time_utils import timestamp_to_seconds

def get_metrics(data, keyword=None):
    ms = data["entity"].keys()
    if keyword is None:
        return ms
    else:
        return [m for m in ms if keyword in m]

def get_metric(data, metric, time_in_sec=False, time_incremental=False):
    try: 
        epochs = eval(data["entity"][metric]["prov-ml:metric_epoch_list"])
        values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
        times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    except: 
        return pd.DataFrame(columns=["epoch", "value", "time"])
    
    # convert to minutes and sort
    if time_in_sec:
        times = [timestamp_to_seconds(ts) for ts in times]
        
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times}).drop_duplicates()

    if time_incremental: 
        df["time"] = df["time"].diff().fillna(0)

    df = df.sort_values(by="time")
    return df

def get_metric_numpy(data, metric, time_in_sec=False, time_incremental=False):
    try: 
        epochs = np.array(json.loads(data["entity"][metric]["prov-ml:metric_epoch_list"]), dtype='i4')
        values = np.array(json.loads(data["entity"][metric]["prov-ml:metric_value_list"]), dtype='f4')
        times = np.array(json.loads(data["entity"][metric]["prov-ml:metric_timestamp_list"]), dtype='i8')
    except Exception as e: 
        print('Impossibile ottenere metriche per il campo: ' + metric)
        print('Errore:', e)
        exit()

    # convert to seconds
    if time_in_sec:
        times = times // 1000
        
    # Sort
    sort_indexes = np.argsort(times)
    times = times[sort_indexes]
    epochs = epochs[sort_indexes]
    values = values[sort_indexes]

    # Calculate incremntal time between steps
    if time_incremental: 
        times = np.diff(times)
        times = np.insert(times, 0, 0)

    return [epochs, values, times, len(epochs)]

def get_avg_metric(data, metric):
    values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
    return sum(values) / len(values)

def get_sum_metric(data, metric):
    values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
    return sum(values)

def get_metric_time(data, metric, time_in_sec=False): 
    times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    if time_in_sec:
        times = [timestamp_to_seconds(ts) for ts in times]
    return max(times) - min(times)


def get_param(data, param):
    return float(data["entity"][param]["prov-ml:parameter_value"])