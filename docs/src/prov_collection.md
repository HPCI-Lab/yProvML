
# Provenance Collection Creation

<div style="display: flex; align-items: center; background-color: #cc3300; color: #333; border: 5px solid #cc3300; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">‼</span>
    <span style="margin-left: 35px; padding: 5px; background-color: white; border-radius: 5px; width: 100%">This feature is temporarily removed</span>
</div>

The provenance collection functionality can be used to create a summary file linking all PROV-JSON files generated during a run. These files come from distributed execution, where each process generates its own log file, and the user may want to create a single file containing all the information.

The collection can be created with the following command: 

```bash
python -m prov4ml.prov_collection --experiment_path experiment_path --output_dir output_dir
```

Where `experiment_path` is the path to the experiment directory containing all the PROV-JSON files, and `output_dir` is the directory where the collection file will be saved. 

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="logging.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="metrics.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>

