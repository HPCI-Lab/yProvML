
# General Logging

When logging parameters and metrics, the user must specify the context of the information. 
The available contexts are: 
 - `TRAINING`: adds the information to the training context  
 - `VALIDATION`: adds the information to the validation context
 - `TESTING`: adds the information to the testing context
 - `MODELS`: adds the information to the models group
 - `DATASETS`: adds the information to the datasets group


The user can easily create new contexts: 

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
prov4ml.create_context("TRAINING_LOD2", prov4ml.Context.TRAINING)
prov4ml.create_context("TRAINING_LOD3", prov4ml.Context.TRAINING_LOD2)

prov4ml.log_param("loss_fn", "MSELoss", prov4ml.Context.TRAINING_LOD2)

<hr style="border: 2px solid #009B77; margin: 20px 0;">


## Log Parameters

To specify arbitrary training parameters used during the execution of the experiment, the user can call the following function. 
    
```python
prov4ml.log_param(
    key: str, 
    value: str, 
    context : Optional[Context] = None
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the parameter |
| `value` | `string` | **Required**. Value of the parameter |
| `context` | `Optional[Context]` | **Optional**. Indicates which context to add the parameter to |


## Log Metrics

To specify metrics, which can be tracked during the execution of the experiment, the user can call the following function.

```python
prov4ml.log_metric(
    key: str, 
    value: float, 
    context: Optional[Context] = None, 
    step: Optional[int] = None, 
    source: Optional[LoggingItemKind] = None, 
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the metric |
| `value` | `float` | **Required**. Value of the metric |
| `context` | `Optional[prov4ml.Context]` | **Required**. Context of the metric |
| `step` | `Optional[int]` | **Optional**. Step of the metric |
| `source` | `Optional[LoggingItemKind]` | **Optional**. Source of the metric |

The *step* parameter is optional and can be used to specify the current time step of the experiment, for example the current epoch, it defaults to 0.
In a similar manner, the *context* parameter can also be omitted, and it will default to the main experiment context. 
The *source* parameter is optional and can be used to specify the source of the metric, so for example which library the data comes from. If omitted, yProv4ML will try to automatically determine the origin. 

## Log Artifacts

To log artifacts, the user can call the following function.

```python
prov4ml.log_artifact(
    artifact_name : str, 
    artifact_path : str, 
    context: Optional[Context] = None,
    step: Optional[int] = None, 
    log_copy_in_prov_directory : bool = True, 
    is_model : bool = False, 
    is_input : bool = False, 
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `artifact_name` | `string` | **Required**. Label to give to the artifact |
| `artifact_path` | `string` | **Required**. Path to the artifact |
| `context` | `Optional[prov4ml.Context]` | **Required**. Context of the artifact |
| `step` | `Optional[int]` | **Optional**. Step of the artifact |
| `log_copy_in_prov_directory` | `bool` | **Optional**. Copies file in artifact directory |
| `is_input` | `bool` | **Optional**. Indicates that the artifact is used as input to the training process. |

The function logs the artifact in the current experiment. The artifact can be a file or a directory. 
All logged artifacts are saved in the artifacts directory of the current experiment, while the related information is saved in the PROV-JSON file, along with a reference to the file. 
The *value* parameter can be any artifact, a file, a path, a value. yProv4ML identifies the correct way to store this parameter in memoty and connect it to the provenance file. 
If *log_copy_in_prov_directory* is `True`, the file at the specified value parameter is copied inside the artefacts directory.  
If *is_input* is `True`, the artifacts will be referenced as such in the W3C prov standard. An example of this would be pretrained model weights. 

## Log Models

```python
prov4ml.log_model(
    model_name: str, 
    model: Union[torch.nn.Module, Any], 
    log_model_info: bool = True, 
    log_model_layers : bool = False,
    is_input: bool = False,
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_name` | `string` | **Required**. Name of the model |
| `model` | `Union[torch.nn.Module, Any]` | **Required**. The model to be logged |
| `log_model_info` | `bool` | **Optional**. Whether to log model information |
| `log_model_layers` | `bool` | **Optional**. Whether to log model layers |
| `is_input` | `bool` | **Optional**. Indicates that the model is used as input to the training process |

It sets the model for the current experiment. It can be called anywhere before the end of the experiment. 
The same call also logs some model information, such as the number of parameters and the model architecture memory footprint. 
The saving of these information can be toggled with the ```log_model_info = False``` parameter. 
The model layers details can be logged in an external .json file, which will be linked to the provenance file as an artefact. 
The parameters saved for each layer depend on the type of the latter, but generally include input and output size, as well as dtype. 

```python
prov4ml.save_model_version(
    model_name: str, 
    model: Union[torch.nn.Module, Any], 
    context: Optional[Context] = None, 
    step: Optional[int] = None, 
    incremental : bool = True, 
    is_input : bool =False, 
)
```

The save_model_version function saves the state of a PyTorch model and logs it as an artifact, enabling version control and tracking within machine learning experiments.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_name`	| `str`|	**Required**. The name under which to save the model. | 
| `model`	| `torch.nn.Module` |	**Required**. The PyTorch model to be saved. |
| `context`	| `Optional[Context]` |	**Optional**. The context in which the model is saved. |
| `step`	| `Optional[int]` |	**Optional**. The step or epoch number associated with the saved model. |
| `incremental`	| `bool` |	**Optional**. Indicates whether there will be multiple versions of this model. |
| `is_input`	| `bool` |	**Optional**. Indicates that the model is used as input to the training process. |

This function saves the model's state dictionary to a specified directory and logs the saved model file as an artifact for provenance tracking. It ensures that the directory for saving the model exists, creates it if necessary, and uses the `torch.save` method to save the model. It then logs the saved model file using `log_artifact`, associating it with the given context and optional step number. 
If ```save_model_version``` is called several times with `incremental = True`, yProv4ML creates an incremental id for each model variation, and saves all in a sub-directory. 

## Log Datasets

yProv4ML offers helper functions to log information and stats on specific datasets.  

```python
prov4ml.log_dataset(
    dataset_label : str, 
    dataset : Union[DataLoader, Subset, Dataset], 
    log_dataset_info : bool = True
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `dataset_label` | `string` | **Required**. The label of the dataset |
| `dataset` | `Union[DataLoader, Subset, Dataset]` | **Required**. The dataset to be logged |
| `log_dataset_info` | `bool` | **Optional**. Whether to log the dataset information |

The function logs the dataset in the current experiment. The dataset can be a DataLoader, a Subset, or a Dataset class from pytorch.
Parameters which are logged include batch size, number of workers, whether the dataset is shuffled, the number of batches and the number of total samples. 

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="prov_graph.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="prov_collection.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>