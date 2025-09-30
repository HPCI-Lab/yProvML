import tensorflow as tf
import prov4ml

class MetricLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, 
            log_every_batch=True, 
            log_every_epoch=False, 
            log_system_metrics=False, 
            log_carbon_metrics=False
            ):
        
        super().__init__()
        self.log_every_batch=log_every_batch 
        self.log_every_epoch=log_every_epoch
        self.log_system_metrics=log_system_metrics 
        self.log_carbon_metrics=log_carbon_metrics
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch
        if not self.log_every_epoch: 
            return
        
        logs = logs or {}
        for k, v in logs.items(): 
            prov4ml.log_metric(k, v, context=prov4ml.Context.TRAINING)

        if self.log_system_metrics: 
            prov4ml.log_system_metrics(context=prov4ml.Context.TRAINING, step=epoch)
        if self.log_carbon_metrics: 
            prov4ml.log_carbon_metrics(context=prov4ml.Context.TRAINING, step=epoch)

    def on_train_batch_end(self, batch, logs=None):
        if not self.log_every_batch: 
            return
        
        logs = logs or {}
        for k, v in logs.items(): 
            prov4ml.log_metric(k, v, context=prov4ml.Context.TRAINING)

        if self.log_system_metrics: 
            prov4ml.log_system_metrics(context=prov4ml.Context.TRAINING, step=self.current_epoch)
        if self.log_carbon_metrics: 
            prov4ml.log_carbon_metrics(context=prov4ml.Context.TRAINING, step=self.current_epoch)