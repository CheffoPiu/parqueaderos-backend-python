import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def entrenar_modelo(df_modelo, guardar_como="modelo_volvera.joblib"):
    # Features y target
    X = df_modelo[["frecuencia", "recencia", "monto_total"]]
    y = df_modelo["volvera"]

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test)
    print("\n📊 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    joblib.dump(modelo, guardar_como)
    print(f"\n✅ Modelo guardado como: {guardar_como}")
