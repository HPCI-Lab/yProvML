

# Execution Time

```python
prov4ml.log_current_execution_time(
    label: str, 
    context: Context, 
    step: Optional[int] = None
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the code portion |
| `context` | `prov4ml.Context` | **Required**. Context of the metric |
| `step` | `int` | **Optional**. Step of the metric |

The `log_current_execution_time` function logs the current execution time of the code portion specified by the label.
The `log_execution_start_time` function logs the start time of the current execution. 
It is automatically called at the beginning of the experiment.

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
prov4ml.log_execution_start_time()
# run training process or other very important tasks...
prov4ml.log_execution_end_time()
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">

The *log_execution_end_time* function logs the end time of the current execution.
It is automatically called at the end of the experiment.


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="system.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="registering_metrics.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>