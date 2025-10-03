
# Register Metrics for custom Operations

<div style="display: flex; align-items: center; background-color: #cc3300; color: #333; border: 5px solid #cc3300; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">‼</span>
    <span style="margin-left: 35px; padding: 5px; background-color: white; border-radius: 5px; width: 100%">This feature is temporarily removed</span>
</div>

After collection of a specific metric, it's very often the case that a user may want to aggregate that information by applying functions such as mean, standard deviation, or min/max. 

yProv4ML allows to register a specific metric to be aggregated, using the function: 

```python
prov4ml.register_final_metric(
    metric_name : str,
    initial_value : float,
    fold_operation : FoldOperation
) 
```

where `fold_operation` indicates the function to be applied to the data. 

Several FoldOperations are already defined, such as MAX, MIN, ADD and SUBRACT. 
In any case the user is always able to define its own custom function, by either defining one with signature: 

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
def custom_foldOperation(x, y): 
    return x // y
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">

Or by passing a lambda function: 

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
prov4ml.register_final_metric("my_metric", 0, lambda x, y: x // y) 
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">

The output of the aggregated metric is saved in the PROV-JSON file, as an attribute of the current execution. 


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="time.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="reproducibility.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>