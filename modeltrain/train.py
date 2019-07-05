from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import os
from weights import get_class_weights
from utility import get_sample_counts
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
from model import ModelFactory
import json

output_dir = "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\modeltrain\\output"
train_dir = (
    "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\cats_and_dogs_small\\train"
)
validation_dir = (
    "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\cats_and_dogs_small\\validation"
)
output_weights_path = os.path.join(output_dir, "weight.h5")
class_names = ["dog", "cat"]

model_factory = ModelFactory()
model = model_factory.get_model(class_names)
print(model.summary())
print(len(model.layers))

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=False,
    height_shift_range=0.05,
    width_shift_range=0.1,
    rotation_range=5,
    shear_range=0.1,
    fill_mode="reflect",
    zoom_range=0.15,
    rescale=1.0 / 255,
)
train_generator = (
    train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
    )
    if len(class_names) > 2
    else train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
    )
)
label_map = train_generator.class_indices
with open("./label_map.json", "w") as f:
    json.dump(label_map, f)
    print("save label map...")
validation_datagen = ImageDataGenerator(
    # samplewise_center=True,
    # samplewise_std_normalization=True,
    # horizontal_flip=True,
    # vertical_flip=False,
    # height_shift_range=0.05,
    # width_shift_range=0.1,
    # rotation_range=5,
    # shear_range=0.1,
    # fill_mode="reflect",
    # zoom_range=0.15,
    rescale=1.0
    / 255
)
validation_generator = (
    validation_datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
    )
    if len(class_names) > 2
    else validation_datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
    )
)

gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
if gpus > 1:
    print(f"** multi_gpu_model is used! gpus={gpus} **")
    model_train = multi_gpu_model(model, gpus)
    # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
    checkpoint = MultiGPUModelCheckpoint(filepath=output_weights_path, base_model=model)
else:
    model_train = model
    checkpoint = ModelCheckpoint(
        output_weights_path, save_weights_only=True, save_best_only=True, verbose=1
    )

optimizer = Adam(lr=0.001)
if len(class_names) > 2:
    model_train.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"]
    )
else:
    model_train.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"]
    )

auroc = MultipleClassAUROC(
    sequence=validation_generator,
    class_names=class_names,
    weights_path=output_weights_path,
    stats={},
    workers=8,
)
callbacks = [
    checkpoint,
    TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=32),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_lr=1e-8
    ),
    # auroc,
    CSVLogger(os.path.join(output_dir, "training_log.csv")),
]
# train_counts, train_pos_counts = get_sample_counts(output_dir, "train", class_names)
print("** compute class weights from training data **")
# class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)
print("** class_weights **")
history = model_train.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator) // 32,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator) // 32,
    callbacks=callbacks,
    # class_weight=class_weights,
    workers=8,
    shuffle=False,
)

