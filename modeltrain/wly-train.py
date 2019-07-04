from keras.applications import densenet
import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import os
from weights import get_class_weights
from utility import get_sample_counts
from keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
    MultipleClassAUROC,
    MultiGPUModelCheckpoint,
)


class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3), module_name="densenet", last_conv_layer="bn"
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(
        self,
        class_names,
        model_name="DenseNet121",
        use_base_weights=True,
        weights_path=None,
        input_shape=None,
    ):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self.models_[model_name]['module_name']}"
            ),
            model_name,
        )

        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg",
        )
        x = base_model.output
        predictions = (
            Dense(len(class_names), activation="softmax", name="predictions")(x)
            if len(class_names) > 2
            else Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        )
        model = Model(inputs=img_input, outputs=predictions)

        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model


model_factory = ModelFactory()
class_names = ["A", "B", "C"]
model = model_factory.get_model(class_names)
print(model.summary())

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
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
)

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
    rescale=1.0 / 255,
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
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
    model_train.compile(optimizer=optimizer, loss="categorical_crossentropy")
else:
    model_train.compile(optimizer=optimizer, loss="binary_crossentropy")

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
        monitor="val_loss", factor=0.1, patience=1, verbose=1, mode="min", min_lr=1e-8
    ),
    auroc,
]
train_counts, train_pos_counts = get_sample_counts(output_dir, "train", class_names)
print("** compute class weights from training data **")
class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)
print("** class_weights **")
history = model_train.fit_generator(
    generator=train_generator,
    # steps_per_epoch=train_steps,
    epochs=100,
    validation_data=validation_generator,
    # validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weights,
    workers=8,
    shuffle=False,
)

