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

outputDir = "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\modeltrain\\output"
trainDir = "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\cats_and_dogs_small\\train"
validationDir = (
    "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\cats_and_dogs_small\\validation"
)
outputWeightsPath = os.path.join(outputDir, "weight.h5")

imageSize = 224
batchSize = 32
classNum = len(os.listdir(trainDir))

models = ModelFactory()

trainDatagen = ImageDataGenerator(
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
trainGenerator = (
    trainDatagen.flow_from_directory(
        trainDir,
        target_size=(imageSize, imageSize),
        batch_size=batchSize,
        class_mode="categorical",
    )
    if classNum > 2
    else trainDatagen.flow_from_directory(
        trainDir,
        target_size=(imageSize, imageSize),
        batch_size=batchSize,
        class_mode="binary",
    )
)

labelMap = trainGenerator.class_indices
classNames = labelMap.keys()
with open(os.path.join(outputDir, "label_map.json"), "w") as f:
    json.dump(labelMap, f)
    print("save label map...")

train_counts, train_pos_counts = get_sample_counts(trainDir, "train", classNames)
print("** compute class weights from training data **")
classWeights = get_class_weights(train_counts, train_pos_counts, multiply=1)
print("** class_weights **")

validationDatagen = ImageDataGenerator(rescale=1.0 / 255)
validationGenerator = (
    validationDatagen.flow_from_directory(
        validationDir,
        target_size=(imageSize, imageSize),
        batch_size=batchSize,
        class_mode="categorical",
    )
    if classNum > 2
    else validationDatagen.flow_from_directory(
        validationDir,
        target_size=(imageSize, imageSize),
        batch_size=batchSize,
        class_mode="binary",
    )
)

model = models.get_model(classNames)
print(model.summary())
print(len(model.layers))

gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
if gpus > 1:
    print(f"** multi_gpu_model is used! gpus={gpus} **")
    model_train = multi_gpu_model(model, gpus)
    # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
    checkpoint = MultiGPUModelCheckpoint(filepath=outputWeightsPath, base_model=model)
else:
    model_train = model
    checkpoint = ModelCheckpoint(
        outputWeightsPath, save_weights_only=True, save_best_only=True, verbose=1
    )

optimizer = Adam(lr=0.001)
if len(classNames) > 2:
    model_train.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"]
    )
else:
    model_train.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"]
    )

callbacks = [
    checkpoint,
    # TensorBoard(log_dir=os.path.join(outputDir, "logs"), batch_size=32),
    ReduceLROnPlateau(
        monitor="val_acc", factor=0.2, patience=5, verbose=1, mode="max", min_lr=1e-8
    ),
    CSVLogger(os.path.join(outputDir, "training_log.csv")),
]

history = model_train.fit_generator(
    generator=trainGenerator,
    steps_per_epoch=len(trainGenerator),
    epochs=100,
    validation_data=validationGenerator,
    validation_steps=len(validationGenerator),
    callbacks=callbacks,
    class_weight=classWeights,
    workers=8,
    shuffle=False,
)

