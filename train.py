import numpy as np
from model import build_model

# dummy data (replace later)
X = np.random.rand(100, 5)
y = np.random.rand(100, 1)

model = build_model(5)
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10)
