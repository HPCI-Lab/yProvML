
# Reproducing Experiments using Provenance Files

<div style="display: flex; align-items: center; background-color: #cc3300; color: #333; border: 5px solid #cc3300; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">‼</span>
    <span style="margin-left: 35px; padding: 5px; background-color: white; border-radius: 5px; width: 100%">This feature is still under development</span>
</div>


With workflow streamlined by yProv4ML, it is trivial to guarantee reproducibility of experiments even just sharing a single provenance file. 
To guarantee the necessary amount of information are present in the prov.json file however, some calls to the library have to be executed. 

```python
def log_execution_command(cmd : str): ...
```

Simply logs the execution command for it to be retrieved by the reproduction script later. This is often a call to python3

```python
def log_source_code(path: Optional[str] = None): ...
```

Logs as an artifact a path to the source code. This could be a single python file (e.g. main.py), a repository link (if the path is not specified in the arguments), or an entire directory of source files. 
In case the source code is not on github, the source files are all copied inside the artifacts directory, and the path logged inside the provenance file will reference whis copy. 

```python
def log_input(inp : Any, log_copy_in_prov_directory : bool = True): ...
```

The directive `log_input` saves with incremental ids a various number of inputs, which are user defined. 
The `log_copy_in_prov_directory` parameter indicates whether the input (if a file) is copied as artifact into the artifact directory. This is necessary for reproducibility of the experiment if the input is not retrievable in any other way. 

```python
def log_output(out : Any, log_copy_in_prov_directory : bool = True): ...
```

The directive `log_output` saves with incremental ids a various number of outputs, which are user defined. 

<div style="display: flex; align-items: center; background-color: #ffcc00; color: #333; border: 5px solid #ffcc00; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">⚠</span>
    <span style="margin-left: 55px; padding: 5px; background-color: white; border-radius: 5px; width:100%">
    Currently, the order of logging of inputs and outputs does matter, which means that concurrent executions will have to log information in order, for a run to be considered reproducible. 
    </span>
</div>



<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="registering_metrics.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="prov_viewer.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>