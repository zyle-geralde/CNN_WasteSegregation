import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from PIL import Image

#load and prepare dataset (2 splits only)
dataset = load_dataset("nflechas/recycling_app", "full")
dataset["test"] = concatenate_datasets([dataset["validation"], dataset["test"]])
del dataset["validation"]

IMG_SIZE = (224, 224)
NUM_CLASSES = dataset["train"].features["objects"].feature["category"].num_classes

def preprocess_data(split):
    images, labels = [], []
    for item in dataset[split]:
        img = item["image"].resize(IMG_SIZE).convert("RGB")
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)

        # Use most frequent label if multiple objects
        categories = item["objects"]["category"]
        if not categories:
            continue
        most_common_label = max(set(categories), key=categories.count)

        images.append(img_array)
        labels.append(most_common_label)

    return np.array(images), to_categorical(np.array(labels), num_classes=NUM_CLASSES)

x_train, y_train = preprocess_data("train")
x_test, y_test = preprocess_data("test")

#build MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
history = model.fit(
    x_train, y_train,
    validation_split=0.1,  #10% of train as validation
    epochs=20,
    batch_size=32,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

#evaluate model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"\nFinal Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

#classification reports
y_train_pred = model.predict(x_train).argmax(axis=1)
y_test_pred = model.predict(x_test).argmax(axis=1)

y_train_true = y_train.argmax(axis=1)
y_test_true = y_test.argmax(axis=1)

print("\nClassification Report (Training):")
print(classification_report(y_train_true, y_train_pred))

print("\nClassification Report (Test):")
print(classification_report(y_test_true, y_test_pred))

#confusion matrices
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

plot_confusion(y_train_true, y_train_pred, "Training Confusion Matrix")
plot_confusion(y_test_true, y_test_pred, "Test Confusion Matrix")

#accuracy and loss plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
