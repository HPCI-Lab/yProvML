
import os
import prov
import prov.model as prov
import pandas as pd

from prov4ml.constants import PROV4ML_DATA

def create_prov_document() -> prov.ProvDocument:
    
    doc = PROV4ML_DATA.root_provenance_doc
    # run_activity = get_activity(doc, "context:" + PROV4ML_DATA.EXPERIMENT_NAME)

    for (name, ctx) in PROV4ML_DATA.metrics.keys():
        metric_file_path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, name + "_" + str(ctx) + f"_GR{PROV4ML_DATA.global_rank}" +".csv")
        data = pd.read_csv(metric_file_path, sep=PROV4ML_DATA.TMP_SEP)

        source = data.columns[2]
        metric_epoch_data = {}
        for _, line in data.iterrows():
            epoch, value, timestamp = line.iloc[0], line.iloc[1], line.iloc[2]
            timestamp = int(timestamp)
            if int(epoch) not in metric_epoch_data:
                metric_epoch_data[epoch] = []
            metric_epoch_data[epoch].append((value, timestamp))

        epochs, values, timestamps = [], [], []
        for epoch, item_ls in metric_epoch_data.items():
            for (val, time) in item_ls:
                epochs.append(epoch)
                values.append(val)
                timestamps.append(time)

        path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR,f"{name}_{str(ctx)}.csv")
        pd.concat(map(lambda x: pd.Series(x), [epochs, values, timestamps])).to_csv(path)
        metric_name = "_".join(path.removesuffix(".csv").split("/")[-1].split("_")[:-1])
        e = PROV4ML_DATA.add_artifact(metric_name,path,0,ctx, is_input=False, log_copy_in_prov_directory=False)
        
        e.add_attributes({
            # 'prov-ml:metric_epoch_list': str(epochs), 
            # 'prov-ml:metric_value_list': str(values),
            # 'prov-ml:metric_timestamp_list': str(timestamps),
            'prov-ml:context': str(ctx),
            'prov-ml:source': str(source)
        })

    return doc