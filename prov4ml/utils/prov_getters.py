
import pandas as pd
from prov4ml.utils.time_utils import timestamp_to_seconds

def get_metrics(data, keyword=None):
    ms = data["entity"].keys()
    if keyword is None:
        return ms
    else:
        return [m for m in ms if keyword in m]

def get_metric(data, metric, time_in_sec=False, time_incremental=False, start_at=None, end_at=None):
    try:
        epochs = eval(data["entity"][metric]["prov-ml:metric_epoch_list"])
        values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
        times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])

        start_at = 0 if start_at is None else start_at
        end_at = len(epochs) if end_at is None else end_at        

        epochs = epochs[start_at:end_at]
        values = values[start_at:end_at]
        times = times[start_at:end_at]
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