import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import models, layers, optimizers, losses, metrics
from imblearn.over_sampling import SMOTE

def get_splitted_data(dataset: np.ndarray, labels: np.ndarray, train_index: np.ndarray, test_index: np.ndarray) -> tuple:
    x_train, x_test = dataset[train_index], dataset[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    return x_train, y_train, x_test, y_test

def get_compiled_model() -> models.Sequential:
    METRICS = [
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.SpecificityAtSensitivity(0, name='specificity'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR'), 
    ]
    
    model = models.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.7),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.6),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.6),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
         optimizers.Adam(lr = 0.001),
         loss = losses.BinaryCrossentropy(),
         metrics = METRICS
    )
    return model

def fit_model(model: models.Sequential, x_train: np.ndarray, y_train: np.ndarray):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    x_train, y_train = SMOTE(sampling_strategy=.2).fit_resample(x_train, y_train.ravel())
    
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    
    history = model.fit(
        x_train, y_train,
        epochs=11,
        batch_size=1024,
        validation_data=(x_val, y_val),
        class_weight=class_weights
    )
    return history

def get_test_results(model, x_test: np.ndarray, y_test: np.ndarray):
    y_pred, y_true = np.round(model.predict(x_test)), y_test.flatten()
    evaluation = model.evaluate(x_test, y_test)
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=['Legitimate', 'Fradulent']).plot()
    return evaluation
