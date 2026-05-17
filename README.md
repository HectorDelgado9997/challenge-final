# 📈 Monthly Investment Recommendation — Challenge Final

[![Status](https://img.shields.io/badge/Status-Completado-brightgreen)](https://github.com/HectorDelgado9997/challenge-final)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue)](https://scikit-learn.org/)
[![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-Optimization-green)](https://pyportfolioopt.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![CI](https://github.com/HectorDelgado9997/challenge-final/actions/workflows/ci.yml/badge.svg)](https://github.com/HectorDelgado9997/challenge-final/actions/workflows/ci.yml)
## 📌 Descripcion

Sistema local de recomendacion de inversion mensual que descarga datos
historicos de **Yahoo Finance**, entrena modelos de Machine Learning para
clasificacion y regresion, y genera una asignacion optima de capital entre
activos financieros usando **Teoria Moderna de Portafolios**.

> ⚠️ Este proyecto es de uso academico y tecnico exclusivamente.
> No constituye asesoramiento financiero.

---

## 🎯 Objetivo

Recomendar como distribuir un monto mensual de inversion entre un universo
de activos financieros, combinando:

- **Clasificacion**: predecir si cada activo superara al benchmark SPY el
  proximo mes
- **Regresion**: predecir el retorno esperado mensual de cada activo
- **Optimizacion**: maximizar el Sharpe Ratio del portafolio resultante

---

## 🗂️ Universo de Activos

| Asset     | Tipo                         |
|-----------|------------------------------|
| SPY       | US Large Cap Equity ETF      |
| QQQ       | US Technology Equity ETF     |
| BTC-USD   | Bitcoin / Criptomoneda       |
| XLE       | US Energy Sector ETF         |
| GRID      | Clean Energy ETF             |
| FLKR      | Frontier Markets ETF         |
| GLD       | Gold ETF                     |
| EWW       | Mexico Equity ETF            |
| EWZ       | Brazil Equity ETF            |

---

## 📁 Estructura del Repositorio

```text
challenge-final/
├── data/
│   ├── raw/                          ← Precios mensuales descargados
│   ├── processed/                    ← Dataset con features y target
│   └── outputs/                      ← Asignacion recomendada
├── docs/
│   ├── dataset_extraction.md
│   ├── model_construction.md
│   ├── mlops.md
│   ├── execution_guide.md
│   └── architecture.md
├── notebooks/                        ← Analisis exploratorio
├── reports/
│   ├── figures/                      ← Graficas de modelos
│   └── metrics/                      ← model_metrics.json
├── scripts/
│   ├── 01_extract_data.py            ← Extraccion de yfinance
│   ├── 02_build_dataset.py           ← Feature engineering
│   ├── 03_train_models.py            ← Entrenamiento + MLflow
│   ├── 04_optimize_portfolio.py      ← Optimizacion de portafolio
│   └── 05_run_full_pipeline.py       ← Pipeline completo
├── src/
│   ├── config/                       ← settings.py
│   ├── data/                         ← extract, build, validate
│   ├── features/                     ← feature_engineering
│   ├── models/                       ← train, evaluate, predict
│   ├── portfolio/                    ← optimize_portfolio
│   ├── mlops/                        ← mlflow_tracking
│   └── utils/                        ← logger, exceptions
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🤖 Modelos

### Clasificacion — ¿Supera al benchmark SPY?

| Modelo              | Pipeline                              |
|---------------------|---------------------------------------|
| Logistic Regression | Imputer → StandardScaler → LogReg     |
| Random Forest       | Imputer → RandomForestClassifier      |

| Metrica      | Descripcion                          |
|--------------|--------------------------------------|
| Precision    | Exactitud en predicciones positivas  |
| Recall       | Cobertura de casos positivos reales  |
| F1 Score     | Media harmonica precision/recall     |
| ROC AUC      | Capacidad discriminativa del modelo  |

### Regresion — Retorno esperado mensual

| Modelo            | Pipeline                            |
|-------------------|-------------------------------------|
| Linear Regression | Imputer → StandardScaler → LinReg   |

| Metrica | Descripcion              |
|---------|--------------------------|
| MAE     | Mean Absolute Error      |
| RMSE    | Root Mean Squared Error  |
| R2      | Coeficiente de determinacion |

---

## 📊 Features Utilizadas (12)

`return_1m` · `return_3m` · `return_6m` · `return_12m` ·
`volatility_3m` · `volatility_6m` · `drawdown_3m` · `drawdown_6m` ·
`sharpe_3m` · `sharpe_6m` · `relative_strength_3m` · `relative_strength_6m`

---

## 🏦 Optimizacion de Portafolio

La optimizacion usa **PyPortfolioOpt** con los retornos esperados predichos
por Linear Regression y la matriz de covarianza historica mensual:

| Parametro             | Valor                        |
|-----------------------|------------------------------|
| Estrategia            | Maximum Sharpe Ratio         |
| Fallback              | Minimum Volatility           |
| Peso minimo por activo| 0.00                         |
| Peso maximo por activo| 0.35                         |
| Tasa libre de riesgo  | 0.00                         |

---

## ⚙️ Instalacion y Ejecucion Rapida

```bash
# 1. Clonar el repositorio
git clone git@github.com:HectorDelgado9997/challenge-final.git
cd challenge-final

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# source .venv/bin/activate       # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar MLflow (terminal 1)
mlflow ui --host 127.0.0.1 --port 5000

# 5. Ejecutar el pipeline completo (terminal 2)
python scripts/05_run_full_pipeline.py --amount 10000
```

> Para instrucciones detalladas ver [`docs/execution_guide.md`](docs/execution_guide.md)

---

## 🔁 Pipeline Completo
Validacion de argumentos
│
▼
Extraccion yfinance → data/raw/monthly_prices.csv
│
▼
Feature engineering → data/processed/model_dataset.csv
│
├── Logistic Regression  ─┐
├── Random Forest         ├── MLflow tracking
└── Linear Regression    ─┘
│
▼
Optimizacion portafolio (max Sharpe)
│
▼
data/outputs/recommended_allocation.csv

---

## 📤 Outputs Generados

```text
data/raw/monthly_prices.csv
data/processed/model_dataset.csv
data/outputs/recommended_allocation.csv
reports/figures/confusion_matrix_logistic_regression.png
reports/figures/confusion_matrix_random_forest.png
reports/figures/roc_curve_logistic_regression.png
reports/figures/roc_curve_random_forest.png
reports/metrics/model_metrics.json
```

---

## 🧪 Tests

```bash
pytest -v
```

---

## 📚 Documentacion

| Archivo                      | Contenido                                    |
|------------------------------|----------------------------------------------|
| `docs/dataset_extraction.md` | Extraccion yfinance, features, targets       |
| `docs/model_construction.md` | Modelos, pipelines, metricas, CV             |
| `docs/mlops.md`              | Configuracion MLflow, runs, artefactos       |
| `docs/execution_guide.md`    | Guia paso a paso de instalacion y ejecucion  |
| `docs/architecture.md`       | Arquitectura, capas y flujo completo de datos|

---

## 🛠️ Stack Tecnologico

| Herramienta      | Uso                              |
|------------------|----------------------------------|
| Python 3.9+      | Lenguaje principal               |
| pandas / numpy   | Manipulacion de datos            |
| scikit-learn     | Modelos ML                       |
| yfinance         | Extraccion de datos financieros  |
| PyPortfolioOpt   | Optimizacion de portafolio       |
| MLflow           | Tracking de experimentos         |
| matplotlib       | Visualizaciones                  |
| seaborn          | Visualizaciones estadisticas     |
| joblib           | Serializacion de modelos         |
| pydantic         | Validacion de datos              |
| python-dotenv    | Variables de entorno             |
| pytest           | Pruebas automatizadas            |

---

## 👤 Autor

**Hector Delgado**
[![GitHub](https://img.shields.io/badge/GitHub-HectorDelgado9997-black)](https://github.com/HectorDelgado9997)
