
import tensorflow as tf
import tensorflow_datasets as tfds

import sys
sys.path.append("./")
import yprov4ml

for _ in range(20): 

    yprov4ml.start_run(
        experiment_name="the_nicola_assumption", 
        prov_user_namespace="www.example.org", 
        provenance_save_dir="prov", 
        collect_all_processes=False, 
        csv_separator=";", 
        save_after_n_logs=100, 
        disable_codecarbon=True
    )

    # Seed setting, not sure all of these are necessary
    SEED = 42
    yprov4ml.log_param("seed", SEED)
    # tf.random.set_seed(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)

    # Log source code and main file for reproducible execution
    yprov4ml.log_execution_command("python", "prov4ml_pol.py")
    yprov4ml.log_source_code("examples/prov4ml_pol.py")

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
        # image = tf.image.resize(image, (32, 32))
        # image = tf.repeat(image, 3, axis=2)
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

    # m_train = sum([np.average(x) for x, _ in ds_train]) / len(ds_train)
    # m_test = sum([np.average(x) for x, _ in ds_test]) / len(ds_test)
    # print(m_train)
    # print(m_test)

    # print(sum([np.average(x-m_train) for x, _ in ds_train]) / (len(ds_train)-1))
    # print(sum([np.average(x-m_test) for x, _ in ds_test]) / (len(ds_test)-1))
    # exit()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # model = tf.keras.applications.ResNet50(classes=10, weights=None, input_shape=(32, 32,3))
    # total_parameters = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    # print(total_parameters)
    # exit()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn2 = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.00001),
        loss=loss_fn, #[loss_fn, loss_fn2]
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    yprov4ml.log_model("model_Sequential", model)

    EPOCHS = 5
    yprov4ml.log_param("num_epochs", EPOCHS)

    model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
        # Add POL Callback, 
        # you can also specify how many samples it skips
        # Not sure which amount is best, 
        # try to have 10 logs for every epoch maybe
        callbacks=[yprov4ml.POLLoggingCallback("model_label", model, log_every_n_steps=50)], 
    )

    yprov4ml.end_run(
        create_graph=False, 
        create_svg=False, 
        crate_ro_crate=False
    )