# Get Data from Provenance Files

yProv4ml offers a set of directives to easilyy extract the information logged from the provenance.json file. 

<div style="display: flex; align-items: center; background-color: #ffcc00; color: #333; border: 5px solid #ffcc00; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">⚠</span>
    <span style="margin-left: 55px; padding: 5px; background-color: white; border-radius: 5px; width:100%">
    All these functions expect the data to be passed to be a pandas.DataFrame. When using a provenance json file coming from yProv4ML, this can be easily obtained following the example below. 
    </span>
</div>

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
import json
data = json.load(open(path_to_prov_json))     
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">

### Utility Functions

```python 
def get_metrics(data : pd.DataFrame, keyword : Optional[str] = None) -> List[str]
```

The `get_metrics` function retrieves all available metrics from the provided provjson file. If a keyword is specified, it filters the results to include only metrics that match the keyword.

| Parameter         | Type               | Default | Description |
|------------------|--------------------|--------------|-------------|
| `data`          | `pd.DataFrame`      | Required     | The dataset containing metrics. |
| `keyword`       | `Optional[str]`     | `None`       | If provided, filters the metrics to only those containing this keyword. |

```python 
def get_metric(
    data : pd.DataFrame, 
    metric : str, 
    time_in_sec : bool = False, 
    time_incremental : bool = False, 
    sort_by : Optional[str] = None, 
    start_at : Optional[int] = None,
    end_at : Optional[int] = None
) -> pd.DataFrame
```

The `get_metric` function extracts a specific metric from the dataset, with additional options for formatting and filtering:
- It allows conversion of time-based metrics to seconds.
- It can return time-incremental values instead of absolute values.
- Sorting and range selection (start and end points) can be applied.


| Parameter         | Type               | Default | Description |
|------------------|--------------------|--------------|-------------|
| `data`          | `pd.DataFrame`      | Required     | The dataset containing metrics. |
| `metric`        | `str`               | Required     | The specific metric to retrieve. |
| `time_in_sec`   | `bool`              | `False`      | If `True`, converts time-based metrics to seconds. |
| `time_incremental` | `bool`           | `False`      | If `True`, returns incremental values instead of absolute values. |
| `sort_by`       | `Optional[str]`     | `None`       | Sorts the metric values by the specified column. |
| `start_at`      | `Optional[int]`     | `None`       | Filters data to start at this index. |
| `end_at`        | `Optional[int]`     | `None`       | Filters data to end at this index. |

The return value for this function is a dataframe containing the following columns: 
- `value`: contains the metric items 
- `epoch`: contains the corresponding epochs
- `time`: contains the corresponding time steps

```python 
def get_param(data : pd.DataFrame, param : str) -> Any
```

Retrieves a single value corresponding to the given param.
This function is useful when the parameter is expected to have a unique value and the label exactly matches in the prov json file.

```python 
def get_params(data : pd.DataFrame, param : str) -> List[Any]
```

Retrieves a list of values for the given param.
This is useful when multiple values exist for the parameter (for example when marked with an incremental ID) in the provenance json file, allowing further analysis or aggregation.

| Parameter | Type           | Return Type  | Description |
|-----------|--------------|--------------|-------------|
| `data`    | `pd.DataFrame` | - | The dataset containing parameters. |
| `param`   | `str`         | - | The specific parameter to retrieve. |

Let me know if you need further clarification! 🚀

More utility functions are also available: 

```python 
def get_avg_metric(data, metric) -> pd.DataFrame: ...
def get_sum_metric(data, metric) -> pd.DataFrame: ...
def get_metric_time(data, metric, time_in_sec=False) -> pd.DataFrame: ...
```


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="prov_viewer.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="examples.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>