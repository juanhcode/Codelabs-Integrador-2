import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ------------------------------  
# 1. Datos de entrenamiento (XOR)  
# ------------------------------  
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# ------------------------------  
# 2. Definir el modelo  
# ------------------------------  
model = Sequential([
    Input(shape=(2,)),            
    Dense(4, activation="relu"),  
    Dense(1, activation="sigmoid") 
])

# ------------------------------  
# 3. Compilar el modelo  
# ------------------------------  
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------  
# 4. Entrenar el modelo (con tiempo)  
# ------------------------------  
start = time.time()
history = model.fit(
    X, y,
    epochs=5000,
    verbose=0
)
end = time.time()

print(f"\nTiempo de entrenamiento: {end - start:.3f} segundos")

# ------------------------------  
# 5. Evaluar y predecir  
# ------------------------------  
print("\nEvaluación final:")
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Loss: {loss:.3f}, Accuracy: {acc:.3f}")

print("\nPredicciones XOR:")
for a, b in X:
    pred = model.predict(np.array([[a, b]]), verbose=0)
    print(f"{a} XOR {b} = {round(pred.item(), 3)}")
