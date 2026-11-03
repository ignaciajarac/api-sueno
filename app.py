# ============================================================
# app.py - API que usa mi modelo de Regresión Logística
# Make le manda los datos de una persona y la API responde
# en qué categoría de deuda de sueño la clasifica.
# ============================================================

# --- 1. Herramientas que voy a usar ---
from flask import Flask, request, jsonify          # Flask crea la "página web" que recibe los datos
import pandas as pd                                # pandas maneja las tablas de datos
from sklearn.compose import ColumnTransformer      # aplica transformaciones a ciertas columnas
from sklearn.linear_model import LogisticRegression  # el modelo
from sklearn.pipeline import Pipeline              # junta todos los pasos en uno solo
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)   # crea la aplicación

# --- 2. Leer la base limpia y entrenar el modelo ---
# Esto pasa una sola vez, cuando la API se enciende.
df = pd.read_csv("sleep_debt_preparado.csv")
y = df["sleep_debt_category"]                       # lo que quiero predecir
X = df.drop(columns=["sleep_debt_category"])        # los datos de la persona

# Columnas de texto: hay que convertirlas a números
COLUMNAS_TEXTO = ["gender", "occupation_type", "chronotype", "primary_bedtime_app"]
COLUMNAS = list(X.columns)   # lista de los 13 datos que el modelo necesita

# El Pipeline hace 3 pasos en orden, igual que en mi trabajo en Quarto:
modelo = Pipeline([
    # Paso 1: convertir el texto a números (one-hot encoding)
    ("columnas", ColumnTransformer(
        [("texto", OneHotEncoder(drop="first"), COLUMNAS_TEXTO)],
        remainder="passthrough")),                   # las columnas numéricas pasan igual
    # Paso 2: estandarizar (media 0 y desviación 1)
    ("estandarizar", StandardScaler()),
    # Paso 3: el modelo de Regresión Logística
    ("regresion", LogisticRegression(max_iter=1000, random_state=1234)),
])
modelo.fit(X, y)   # entrena el modelo con los datos

# --- 3. Ruta de inicio: sirve para comprobar que la API está viva ---
@app.route("/")
def inicio():
    return "API del modelo de deuda de sueño funcionando. Usa POST /predecir"

# --- 4. Ruta principal: aquí Make envía los datos de una persona ---
@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json(force=True, silent=True)   # lee lo que mandó Make
    if not datos:
        return jsonify(error="Envía un JSON con los datos de la persona"), 400

    # Revisa que no falte ningún dato
    faltan = [c for c in COLUMNAS if c not in datos]
    if faltan:
        return jsonify(error="Faltan campos", campos_faltantes=faltan), 400

    # Arma una tabla de una sola fila con los datos de la persona
    fila = pd.DataFrame([{c: datos[c] for c in COLUMNAS}])

    # Make envía todo como texto, así que convierto los números
    for c in COLUMNAS:
        if c not in COLUMNAS_TEXTO:
            try:
                fila[c] = pd.to_numeric(fila[c])
            except ValueError:
                return jsonify(error=f"El campo {c} debe ser un número"), 400

    # El modelo predice la categoría y las probabilidades
    try:
        categoria = modelo.predict(fila)[0]
        proba = modelo.predict_proba(fila)[0]
    except ValueError as e:
        return jsonify(error="Valor de texto no reconocido", detalle=str(e)), 400

    probabilidades = {c: round(float(p), 3) for c, p in zip(modelo.classes_, proba)}

    # Respuesta que recibe Make
    return jsonify(
        categoria=categoria,                         # la decisión del modelo
        confianza=round(float(proba.max()), 3),      # qué tan seguro está
        probabilidades=probabilidades,               # probabilidad de cada categoría
    )

# --- 5. Para probarla en mi computador (Render no usa esto) ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
