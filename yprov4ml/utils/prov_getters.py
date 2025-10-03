
# KEEP THESE FOR NAN eval() #
import numpy as np
from torch import nan
#############################

import pandas as pd
from yprov4ml.utils.time_utils import timestamp_to_seconds
from typing import Optional

def get_metrics(data, keyword=None):
    ms = data["entity"].keys()
    if keyword is None:
        return ms
    else:
        return [m for m in ms if keyword in m]

def get_metric(data, metric, time_in_sec=False, time_incremental=False, sort_by=None, start_at=None, end_at=None):

    if metric not in data["entity"].keys(): 
        raise AttributeError(f">get_metric({metric}) not found in prov file")

    epochs = eval(data["entity"][metric]["prov-ml:metric_epoch_list"])
    values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
    times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])

    start_at = 0 if start_at is None else start_at
    end_at = len(epochs) if end_at is None else end_at        

    epochs = epochs[start_at:end_at]
    values = values[start_at:end_at]
    times = times[start_at:end_at]
    
    # convert to minutes and sort
    if time_in_sec:
        times = [timestamp_to_seconds(ts) for ts in times]
        
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times})#.drop_duplicates()
    if time_incremental: 
        df["time"] = df["time"].diff().fillna(0)

    if sort_by is not None: 
        df = df.sort_values(by=sort_by)
    
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

def get_params(data, param):
    return [data["entity"][ent]["prov-ml:parameter_value"] for ent in data["entity"].keys() if param in ent]

def get_param(data, param): 
    if param in data["entity"].keys():
        return data["entity"][param]["prov-ml:parameter_value"]
    return None


def get_experiemnt_id(data) -> Optional[str]:
    return list(data["activity"].keys())[0]

def get_execution_command(data) -> Optional[str]:
    return get_param(data, "prov-ml:execution_command")

def get_source_code(data) -> Optional[str]: 
    return get_param(data, "prov-ml:source_code")

def get_inputs(data): 
    return get_params(data, "prov-ml:input")

def get_outputs(data): 
    return get_params(data, "prov-ml:output")
