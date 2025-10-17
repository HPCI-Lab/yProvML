import tensorflow as tf
import yprov4ml

class POLLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_label, model_ptr, log_every_n_steps=5):
        
        super().__init__()
        self.current_epoch = 0
        self.model_label = model_label
        self.model = model_ptr
        self.log_every_n_steps = log_every_n_steps
        self.step_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):        
        logs = logs or {}
        if "loss" not in logs.keys(): 
            raise Exception("A loss is required")
        
        if self.step_counter % self.log_every_n_steps == 0: 
            yprov4ml.log_proof_of_learning_step(self.model_label, self.model, logs["loss"], batch, step=self.current_epoch, context=yprov4ml.Context.TRAINING)
        else: 
            self.step_counter += 1