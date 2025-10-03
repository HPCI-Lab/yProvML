
import tensorflow as tf
import tensorflow_datasets as tfds

import sys
sys.path.append("./")
import yprov4ml

yprov4ml.start_run(
    experiment_name="torchless_run", 
    prov_user_namespace="www.example.org", 
    provenance_save_dir="prov", 
    collect_all_processes=False, 
    save_after_n_logs=100
)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

yprov4ml.log_dataset("ds_train", ds_train)
yprov4ml.log_dataset("ds_test", ds_test)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
yprov4ml.log_model("model_Sequential", model)

EPOCHS = 6
yprov4ml.log_param("num_epochs", EPOCHS)

model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    callbacks=[yprov4ml.MetricLoggingCallback(log_carbon_metrics=False, log_system_metrics=True)], 
)

yprov4ml.end_run(
    create_graph=True, 
    create_svg=True, 
    crate_ro_crate=True
)