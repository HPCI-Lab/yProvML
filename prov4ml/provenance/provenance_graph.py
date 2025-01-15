
import os
import sys
import prov
import prov.model as prov
from datetime import datetime
import getpass
import subprocess
import warnings
import zarr
from numpy import array2string, inf

from prov4ml.constants import PROV4ML_DATA
from prov4ml.datamodel.attribute_type import Prov4MLAttribute
from prov4ml.datamodel.artifact_data import artifact_is_pytorch_model
from prov4ml.provenance.context import Context
from prov4ml.utils.funcs import get_global_rank, get_runtime_type

from contextlib import contextmanager
import time

tempo_totale = 0

@contextmanager
def time_this():
    """
    Utility to time excecution of a block of istructions.

    with time_this():
        instructions
    """
    global tempo_totale
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        tempo_totale += elapsed_time
        # CRED = '\033[91m'
        # CEND = '\033[0m'
        # print(f'Tempo di esecuzione: ' + CRED + f'{elapsed_time:.4f}' + CEND + ' secondi')

def calculate_energy_consumption(
    doc: prov.ProvDocument,
    ctx: Context,
    epochs: list[int],
    timestamps: list[int],
    values: list[float], 
) -> None:
    """
    Calculates energy consumption based on power usage values and updates the provenance document.

    This function computes the total energy consumption from power usage values over specified epochs,
    and records the result in the given provenance document. It also creates or updates entities and
    relationships in the document related to energy consumption metrics.

    Parameters:
    -----------
    doc : prov.ProvDocument
        The provenance document to which the energy consumption data will be added.
    ctx : Context
        The context in which the energy consumption was measured (e.g., TRAINING, VALIDATION, EVALUATION).
    epochs : list[int]
        A list of epochs during which power usage was measured.
    timestamps : list[int]
        A list of timestamps corresponding to the epochs, representing the time at which power usage was recorded.
    values : list[float]
        A list of power usage values recorded during the specified epochs.

    Returns:
    --------
    None

    Notes:
    ------
    - The energy consumption is computed by summing the product of the time interval between successive
      timestamps and the corresponding power usage values.
    - The function creates or updates an entity in the provenance document to represent the energy consumption metric.
    - Relationships are established between the energy consumption entity and the epochs during which the measurements were taken.
    - The provenance document's energy consumption entity is updated with attributes including epochs, values, and timestamps.

    Examples:
    ---------
    calculate_energy_consumption(
        doc=prov_document_instance,
        ctx=Context.TRAINING,
        epochs=[1, 2, 3],
        timestamps=[1000, 2000, 3000],
        values=[150.0, 160.0, 155.0]
    )
    """
    energy = 0
    for i in range(1, len(epochs)):
        energy += (timestamps[i] - timestamps[i-1]) * values[i]

    if not doc.get_record(f'energy_consumption_{ctx}'):
        metric_entity = doc.entity(f'energy_consumption_{ctx}',{
            'prov-ml:type':Prov4MLAttribute.get_attr('Metric'),
            'prov-ml:name':Prov4MLAttribute.get_attr("energy_consumption"),
            'prov-ml:context':Prov4MLAttribute.get_attr(ctx),
            'prov-ml:source':Prov4MLAttribute.get_source_from_kind("gpu_power_usage"),
        })
    else:
        metric_entity = doc.get_record(f'energy_consumption_{ctx}')[0]

    for epoch in epochs: 
        doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',identifier=f'energy_consumption_train_{epoch}_gen')
    
    metric_entity.add_attributes({
        'prov-ml:metric_epoch_list': Prov4MLAttribute.get_attr(epochs), 
        'prov-ml:metric_value_list': Prov4MLAttribute.get_attr(energy),
        'prov-ml:metric_timestamp_list': Prov4MLAttribute.get_attr(timestamps),
        'prov-ml:context': Prov4MLAttribute.get_attr(ctx),
    })
    
def save_metric_from_file(
        metric_file : str,
        name : str, 
        ctx:Context, 
        doc:prov.ProvDocument, 
        run_activity: prov.ProvActivity
    ) -> None:
    """
    Saves metric data from a file to a provenance document.

    This function reads metric data from a file, processes the data, and updates
    the provided `prov.ProvDocument` with entities, activities, and relationships
    related to the metrics. It handles different contexts (training, validation,
    evaluation) and updates the provenance document accordingly.

    Parameters:
    -----------
    metric_file : str
        The name of the file containing the metric data.
    name : str
        The name of the metric.
    ctx : Context
        The context in which the metric was collected (e.g., TRAINING, VALIDATION, EVALUATION).
    doc : prov.ProvDocument
        The provenance document to which the metric data will be added.
    run_activity
        The activity representing the current run or experiment execution.

    Returns:
    --------
    None

    Notes:
    ------
    - The metric file should be formatted with the source on the first line, followed
      by metric data in the format `epoch,value,timestamp` on subsequent lines.
    - The function creates or updates metric entities and activities in the provenance
      document based on the context and metric data.
    - Different activities are created for training, validation, and evaluation contexts.
    - The metric entity's attributes are updated with epoch lists, value lists, and
      timestamps.

    Examples:
    ---------
    save_metric_from_file(
        metric_file='metric_data.txt',
        name='accuracy',
        ctx=Context.TRAINING,
        doc=prov_document_instance,
        run_activity=current_run_activity
    )
    """
    if PROV4ML_DATA.METRICS_FILE_TYPE == "zarr":
        dataset = zarr.open(os.path.join(PROV4ML_DATA.METRICS_DIR, metric_file), 'r')
        source = ', '.join(dataset.attrs.values())

        if not doc.get_record(f'{name}_{ctx}'):
            metric_entity = doc.entity(f'{name}_{ctx}',{
                'prov-ml:type':Prov4MLAttribute.get_attr('Metric'),
                'prov-ml:name':Prov4MLAttribute.get_attr(name),
                'prov-ml:context':Prov4MLAttribute.get_attr(ctx),
                'prov-ml:source':Prov4MLAttribute.get_source_from_kind(source),
            })
        else:
            metric_entity = doc.get_record(f'{name}_{ctx}')[0]

        print(dataset['epochs'].info)

        for epoch in set(dataset['epochs']):
            if ctx == Context.TRAINING: 
                if not doc.get_record(f'epoch_{epoch}'):
                    train_activity=doc.activity(f'epoch_{epoch}',other_attributes={
                        "prov-ml:type": Prov4MLAttribute.get_attr("TrainingExecution")
                    })
                    doc.wasStartedBy(train_activity,run_activity)

                doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',identifier=f'{name}_train_{epoch}_gen')
                
            elif ctx == Context.VALIDATION:
                val_name = f'val_epoch_{epoch}'
                if not doc.get_record(val_name):
                    train_activity=doc.activity(val_name,other_attributes={
                        "prov-ml:type": Prov4MLAttribute.get_attr("ValidationExecution"),
                    })
                    doc.wasStartedBy(train_activity,run_activity)

                doc.wasGeneratedBy(metric_entity,val_name,identifier=f'{name}_val_{epoch}_gen')
                
            elif ctx == Context.EVALUATION:
                if not doc.get_record('test'):
                    eval_activity=doc.activity('test',other_attributes={
                    "prov-ml:type": Prov4MLAttribute.get_attr("TestingExecution")})
                    doc.wasStartedBy(eval_activity,run_activity)

                doc.wasGeneratedBy(metric_entity,'test',identifier=f'test_gen')

        metric_entity.add_attributes({
        # array2string is needed because using simply str() with numpy arrays adds '\n' after a certain number of characters to facilitate console view, and these '\n' were being added to the provenance file
        'prov-ml:metric_epoch_list': Prov4MLAttribute.get_attr(array2string(dataset['epochs'][:], separator=', ', max_line_width=inf)), 
        'prov-ml:metric_value_list': Prov4MLAttribute.get_attr(array2string(dataset['values'][:], separator=', ', max_line_width=inf)),
        'prov-ml:metric_timestamp_list': Prov4MLAttribute.get_attr(array2string(dataset['timestamps'][:], separator=', ', max_line_width=inf)),
        'prov-ml:context': Prov4MLAttribute.get_attr(ctx),
        })
    else:
        with open(os.path.join(PROV4ML_DATA.METRICS_DIR, metric_file), 'r') as f:
            lines = f.readlines()
            source = lines[0].split(',')[2]

        if not doc.get_record(f'{name}_{ctx}'):
            metric_entity = doc.entity(f'{name}_{ctx}',{
                'prov-ml:type':Prov4MLAttribute.get_attr('Metric'),
                'prov-ml:name':Prov4MLAttribute.get_attr(name),
                'prov-ml:context':Prov4MLAttribute.get_attr(ctx),
                'prov-ml:source':Prov4MLAttribute.get_source_from_kind(source),
            })
        else:
            metric_entity = doc.get_record(f'{name}_{ctx}')[0]

        epochs = []
        values = []
        timestamps = []
        for line in lines[1:]:
            epoch, value, timestamp = line.split(',')
            epochs.append(int(epoch))
            values.append(float(value))
            timestamps.append(int(timestamp))

        print(len(epochs))

        for epoch in set(epochs):
            if ctx == Context.TRAINING: 
                if not doc.get_record(f'epoch_{epoch}'):
                    train_activity=doc.activity(f'epoch_{epoch}',other_attributes={
                        "prov-ml:type": Prov4MLAttribute.get_attr("TrainingExecution")
                    })
                    doc.wasStartedBy(train_activity,run_activity)

                doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',identifier=f'{name}_train_{epoch}_gen')
                
            elif ctx == Context.VALIDATION:
                val_name = f'val_epoch_{epoch}'
                if not doc.get_record(val_name):
                    train_activity=doc.activity(val_name,other_attributes={
                        "prov-ml:type": Prov4MLAttribute.get_attr("ValidationExecution"),
                    })
                    doc.wasStartedBy(train_activity,run_activity)

                doc.wasGeneratedBy(metric_entity,val_name,identifier=f'{name}_val_{epoch}_gen')
                
            elif ctx == Context.EVALUATION:
                if not doc.get_record('test'):
                    eval_activity=doc.activity('test',other_attributes={
                    "prov-ml:type": Prov4MLAttribute.get_attr("TestingExecution")})
                    doc.wasStartedBy(eval_activity,run_activity)

                doc.wasGeneratedBy(metric_entity,'test',identifier=f'test_gen')

        metric_entity.add_attributes({
            'prov-ml:metric_epoch_list': Prov4MLAttribute.get_attr(epochs), 
            'prov-ml:metric_value_list': Prov4MLAttribute.get_attr(values),
            'prov-ml:metric_timestamp_list': Prov4MLAttribute.get_attr(timestamps),
            'prov-ml:context': Prov4MLAttribute.get_attr(ctx),
        })

def create_prov_document() -> prov.ProvDocument:
    """
    Generates the first level of provenance for a given run.

    Returns:
        prov.ProvDocument: The provenance document.
    """
    doc = prov.ProvDocument()

    #set namespaces
    doc.set_default_namespace(PROV4ML_DATA.USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('prov-ml', 'prov-ml')

    run_entity = doc.entity(f'{PROV4ML_DATA.EXPERIMENT_NAME}',other_attributes={
        "prov-ml:provenance_path":Prov4MLAttribute.get_attr(PROV4ML_DATA.PROV_SAVE_PATH),
        "prov-ml:artifact_uri":Prov4MLAttribute.get_attr(PROV4ML_DATA.ARTIFACTS_DIR),
        "prov-ml:run_id":Prov4MLAttribute.get_attr(PROV4ML_DATA.RUN_ID),
        "prov-ml:type": Prov4MLAttribute.get_attr("LearningStage"),
        "prov-ml:user_id": Prov4MLAttribute.get_attr(getpass.getuser()),
    })

    # add python version to run entity
    run_entity.add_attributes({"prov-ml:python_version":Prov4MLAttribute.get_attr(sys.version)})

    # check if requirements.txt exists
    # if not os.path.exists("requirements_execution.txt"):
    #     os.popen("pipreqs --force .")

    if os.path.exists("requirements.txt"):
        env_reqs = open("requirements.txt", "r").readlines()
        env_reqs = [req.strip() for req in env_reqs]
        env_reqs = "\n".join(env_reqs)  
        run_entity.add_attributes({"prov-ml:requirements":Prov4MLAttribute.get_attr(env_reqs)})

    global_rank = get_global_rank()
    runtime_type = get_runtime_type()
    if runtime_type == "slurm":
        node_rank = os.getenv("SLURM_NODEID", None)
        local_rank = os.getenv("SLURM_LOCALID", None) 
        run_entity.add_attributes({
            "prov-ml:global_rank":Prov4MLAttribute.get_attr(global_rank),
            "prov-ml:local_rank":Prov4MLAttribute.get_attr(local_rank),
            "prov-ml:node_rank":Prov4MLAttribute.get_attr(node_rank),
        })
    elif runtime_type == "single_core":
        run_entity.add_attributes({
            "prov-ml:global_rank":Prov4MLAttribute.get_attr(global_rank)
        })

    run_activity = doc.activity(f'{PROV4ML_DATA.EXPERIMENT_NAME}_execution', other_attributes={
        'prov-ml:type': Prov4MLAttribute.get_attr("LearningExecution"),
    })
        #experiment entity generation
    experiment = doc.entity(PROV4ML_DATA.EXPERIMENT_NAME,other_attributes={
        "prov-ml:type": Prov4MLAttribute.get_attr("Experiment"),
        "prov-ml:experiment_name": Prov4MLAttribute.get_attr(PROV4ML_DATA.EXPERIMENT_NAME),
    })

    user_ag = doc.agent(f'{getpass.getuser()}')
    doc.wasAssociatedWith(f'{PROV4ML_DATA.EXPERIMENT_NAME}_execution',user_ag)
    doc.entity('source_code',{
        "prov-ml:type": Prov4MLAttribute.get_attr("SourceCode"),
        "prov-ml:source_name": Prov4MLAttribute.get_attr(__file__.split('/')[-1]),
        "prov-ml:runtime_type": Prov4MLAttribute.get_attr(runtime_type),
    })

    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        doc.activity('commit',other_attributes={"prov-ml:source_git_commit": Prov4MLAttribute.get_attr(commit_hash)})
        doc.wasGeneratedBy('source_code','commit')
        doc.wasInformedBy(run_activity,'commit')
    except:
        warnings.warn("Git not found, skipping commit hash retrieval")
        doc.used(run_activity,'source_code')


    doc.hadMember(experiment,run_entity)
    doc.wasGeneratedBy(run_entity,run_activity)
    
    if os.path.exists(PROV4ML_DATA.METRICS_DIR):
        all_metrics = os.listdir(PROV4ML_DATA.METRICS_DIR) 
    else:
        all_metrics = []

    for metric_file in all_metrics:
        # if global_rank is not None:
        name = "_".join(metric_file.split('_')[:-2])
        ctx = metric_file.split('_')[-2].strip()
        # else: 
        # name = "_".join(metric_file.split('_')[:-1])
        # ctx = metric_file.split('_')[-1].replace(".txt","")
        ctx = Context.get_context_from_string(ctx)
        with time_this():
            save_metric_from_file(metric_file, name, ctx, doc, run_activity)

    print(f"Tempo totale: {tempo_totale:.4f} secondi")
                        
    for name, param in PROV4ML_DATA.parameters.items():
        if "dataset" in name: continue

        ent = doc.entity(f'{name}',{
            'prov-ml:parameter_value': Prov4MLAttribute.get_attr(param.value),
            'prov-ml:type': Prov4MLAttribute.get_attr('Parameter'),
        })
        doc.used(run_activity,ent)

    # create entity for final run statistics
    final_run_stats = doc.entity('final_run_statistics',{
        'prov-ml:type': Prov4MLAttribute.get_attr('RunStatistics'),
    })
    other_attributes = {}
    for (name, metric) in PROV4ML_DATA.cumulative_metrics.items():
        other_attributes[f'prov-ml:{name}'] = Prov4MLAttribute.get_attr(metric.current_value)
    final_run_stats.add_attributes(other_attributes)
    doc.wasGeneratedBy(final_run_stats,run_activity)

    #dataset entities generation
    ent_ds = doc.entity(f'datasets')

    for name, param in PROV4ML_DATA.parameters.items():
        if "dataset_stat" in name:
            dataset_name = name.split('_')[0] + "_dataset"
            if not doc.get_record(f'{dataset_name}'):
                ent = doc.entity(f'{dataset_name}',{'prov-ml:type': Prov4MLAttribute.get_attr('Dataset')})

                doc.used(run_activity,ent)
                doc.hadMember(ent_ds,f'{dataset_name}')

            ent = doc.get_record(f'{dataset_name}')[0]

            label = name.split('_')[-1]
            ent.add_attributes({f'prov-ml:{label}': Prov4MLAttribute.get_attr(param.value)})

    doc.wasGeneratedBy(ent_ds,run_activity)

    #model version entities generation
    model_version = PROV4ML_DATA.get_final_model()
    if model_version:
        model_entity_label = model_version.path
        modv_ent=doc.entity(model_entity_label,{
            "prov-ml:type": Prov4MLAttribute.get_attr("ModelVersion"),
            'prov-ml:creation_epoch': Prov4MLAttribute.get_attr(model_version.step),
            'prov-ml:artifact_uri': Prov4MLAttribute.get_attr(model_version.path),
            'prov-ml:creation_timestamp': Prov4MLAttribute.get_attr(datetime.fromtimestamp(model_version.creation_timestamp / 1000)),
            'prov-ml:last_modified_timestamp': Prov4MLAttribute.get_attr(datetime.fromtimestamp(model_version.last_modified_timestamp / 1000)),
        })
        doc.wasGeneratedBy(modv_ent,run_activity,identifier=f'{model_entity_label}_gen')
    
    registration_label = 'prov-ml:ModelRegistration'
    model_ser = doc.activity(registration_label)
    doc.wasInformedBy(model_ser,run_activity)
    if model_version:
        doc.wasGeneratedBy(model_entity_label,model_ser)
    else:
        model_entity_label = registration_label
    
    for artifact in PROV4ML_DATA.get_model_versions()[:-1]: 
        doc.hadMember(model_entity_label,f"{artifact.path}")    

    # doc.activity("data_preparation",other_attributes={"prov-ml:type":Prov4MLAttribute.get_attr("FeatureExtractionExecution")})
    
    #artifact entities generation
    for artifact in PROV4ML_DATA.get_artifacts():
        ent=doc.entity(f'{artifact.path}',{
            'prov-ml:artifact_path': Prov4MLAttribute.get_attr(artifact.path),
        })
        #the FileInfo object stores only size and path of the artifact, specific connectors 
        # to the artifact store are needed to get other metadata
        if artifact_is_pytorch_model(artifact):
            doc.wasGeneratedBy(f"{artifact.path}", model_ser)
        else: 
            doc.wasGeneratedBy(ent,run_activity,identifier=f'{artifact.path}_gen')    

    return doc
