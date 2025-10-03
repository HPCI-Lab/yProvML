
# Setup

Before using the library, the user must set up the yProv4ML execution, as well as library specific configurations: 

```python
prov4ml.start_run(
    experiment_name: str, 
    prov_user_namespace: Optional[str] = None,
    prov_save_path: Optional[str] = None, 
    collect_all_processes: bool = False, 
    save_after_n_logs: int = 100, 
    rank: Optional[int] = None, 
    metrics_file_type: MetricsType = MetricsType.ZARR,
    csv_separator:str = ",", 
    use_compressor: Optional[CompressorType] = None,
)
```

The parameters are as follows:

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `experiment_name` | `string` | **Required**. Name of the experiment |
| `prov_user_namespace` | `string` | **Optional**. User namespace for the provenance graph |
| `prov_save_path` | `string` | **Optional**. Directory to save the provenance graph |
| `collect_all_processes` | `bool` | **Optional**. Whether to collect all processes |
| `save_after_n_logs` | `int` | **Optional**. Save the graph after n logs |
| `rank` | `int` | **Optional**. Rank of the process |
| `metrics_file_type` | `MetricsType` | **Optional**. File format in which to save metrics, can be MetricsType.ZARR, MetricsType.CSV, MetricsType.NETCDF |
| `csv_separator` | `str` | **Optional**. Separator used if `metrics_file_type` is MetricsType.CSV. |
| `disable_codecarbon` | `Optional[bool]` | **Optional**. Whether to use codecarbon to calculate stats. |
| `use_compressor` | `Optional[CompressorType]` | **Optional**. Type of compressor used to create the run zip. If None, then no compression is applied. |

`prov_user_namespace` is a required string that defines the namespace under which all provenance data will be grouped. It helps in logically separating and organizing data across different users or projects, ensuring that the provenance graph remains structured and easily navigable. 

`collect_all_processes`: A boolean flag that, when set to True, enables the collection of provenance data from all processes, which is particularly useful in multi-processing or distributed computing environments. By default, this is False, meaning only the main process (at `rank` 0) will be monitored unless otherwise specified.

`save_after_n_logs`: An optional integer that determines how frequently the provenance graph should be saved based on the number of logs collected. For example, if set to 100, the graph will be saved every 100 logs. This is essentially a caching system, which balances between execution time and RAM usage.

`rank`: This optional integer is used in distributed settings to specify the rank or ID of the currently collecting process. Leaving this parameter empty and `collect_all_processes` to `False` implies that only the process at `rank` 0 will collect data. 

`disable_codecarbon`: An optional boolean that controls whether the CodeCarbon tool is used for tracking the environmental impact of the experiment (e.g., carbon emissions). Setting this to True disables CodeCarbon integration, which can be useful in environments where this measurement is not needed or supported.

---

At the end of the experiment, the user must end the run:

```python
prov4ml.end_run(
    create_graph: Optional[bool] = False, 
    create_svg: Optional[bool] = False, 
    crate_ro_crate: Optional[bool]=False,
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `create_graph` | `Optional[bool]` | **Optional**. Whether to create the graph |
| `create_svg` | `Optional[bool]` | **Optional**. Whether to create the svg |
| `crate_ro_crate` | `Optional[bool]` | **Optional**. Indicates whether to pack the experiment in a ro-crate zip, containing all the run's information. |

This call allows the library to save the provenance graph in the specified directory. 

<div style="display: flex; align-items: center; background-color: #ffcc00; color: #333; border: 5px solid #ffcc00; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">⚠</span>
    <span style="margin-left: 55px; padding: 5px; background-color: white; border-radius: 5px; width:100%">If create_svg is True then create_graph has to be necessairly set to True, as the creation of the former requires the latter. </span>
</div>

We allow to pack the entire experiment data as a [RO-Crate](https://www.researchobject.org/ro-crate/) to enable the user to easily share the final execution of his experiment. 

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="installation.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="prov_graph.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>