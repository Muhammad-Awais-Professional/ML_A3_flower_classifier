import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import joblib
iris = load_iris()
X = iris.data  
y = iris.target  
y_encoded = tf.keras.utils.to_categorical(y, num_classes=3)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
model = Sequential([
    Dense(3, input_shape=(4,), activation='softmax', kernel_regularizer=l2(0.001))
])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,              
    batch_size=16,           
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

model.save('iris_logistic_model.h5')
print("Trained model saved as 'iris_logistic_model.h5'.")

joblib.dump(scaler, 'iris_scaler.pkl')
print("Scaler saved as 'iris_scaler.pkl'.")
