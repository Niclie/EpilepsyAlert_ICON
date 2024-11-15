import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def logistic_regression(training_data, training_label):
    model = LogisticRegression(solver='liblinear')
    model.fit(training_data, training_label)
    
    return model

    
def decision_tree(training_data, training_label):
    model = DecisionTreeClassifier(max_depth=3)    
    model.fit(training_data, training_label)
    
    return model


def cnn(training_data, training_label, out_path, file_name, epochs=500, batch_size=32, early_stopping=50):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input(training_data[0].shape),
            
            tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv1D(filters=4, kernel_size=16, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'{out_path}/{file_name}.keras', save_best_only=True, monitor="val_loss", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience = early_stopping, verbose=1)
    ]

    model.compile(
        optimizer = 'adam',
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )

    history = model.fit(
        training_data,
        training_label,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_split = 0.2,
        verbose = 1
    )

    return history