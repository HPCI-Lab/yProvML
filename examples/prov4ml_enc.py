import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append("./")
import yprov4ml

# Initialize provenance tracking
yprov4ml.start_run(
    experiment_name="selfsupervised_mnist_autoencoder",
    prov_user_namespace="www.example.org",
    provenance_save_dir="prov",
    collect_all_processes=False,
    csv_separator=";",
    save_after_n_logs=100,
    disable_codecarbon=True
)

# Set seed
SEED = 42
tf.random.set_seed(SEED)
yprov4ml.log_param("seed", SEED)

# Log code for reproducibility
yprov4ml.log_execution_command("python", "autoencoder_mnist.py")
yprov4ml.log_source_code("examples/autoencoder_mnist.py")

# Load MNIST data
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

yprov4ml.log_dataset("ds_train", ds_train)
yprov4ml.log_dataset("ds_test", ds_test)

# Normalize & reshape function
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / (255.0 * 1000)
    return image, image  # self-supervised target = input itself

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache().batch(128).prefetch(tf.data.AUTOTUNE)

# --- Define a convolutional autoencoder ---
def build_autoencoder():
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same')
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

    return tf.keras.Model(encoder.input, decoder(encoder.output))

model = build_autoencoder()
model.summary()

# Compile
loss_fn = tf.keras.losses.MeanSquaredError()
# loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss_fn)

yprov4ml.log_model("autoencoder_mnist", model)

# Training parameters
EPOCHS = 5
yprov4ml.log_param("num_epochs", EPOCHS)

# Train the model
model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    callbacks=[
        yprov4ml.POLLoggingCallback("autoencoder", model, log_every_n_steps=5)
    ],
)

# End provenance run
yprov4ml.end_run(
    create_graph=True,
    create_svg=True,
    crate_ro_crate=False
)
