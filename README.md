# HabitAlpes - Modelo de Predicción de Precios de Apartamentos

**Taller 2 - Ciencia de Datos Aplicada**
Universidad de los Andes - MINE-4101

---

## Tabla de Contenidos

- [Resumen del Proyecto](#resumen-del-proyecto)
- [Contexto de Negocio](#contexto-de-negocio)
- [Dataset](#dataset)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instrucciones de Ejecución](#instrucciones-de-ejecución)
- [Resumen de Resultados](#resumen-de-resultados)
- [Autores](#autores)

---

## Resumen del Proyecto

Este proyecto desarrolla un **modelo de machine learning** para predecir precios de venta de apartamentos en Bogotá, Colombia, para **HabitAlpes**, una startup de consultoría inmobiliaria. El modelo busca reducir el tiempo de valoración de expertos de 6 horas a 1 hora por propiedad, manteniendo alta precisión.

### Objetivos

1. **[10%] Entendimiento de Datos**: Analizar 43,013 registros de apartamentos con 46 características
2. **[20%] Desarrollo del Modelo**: Entrenar y comparar múltiples modelos de regresión ML
3. **[20%] Evaluación Cuantitativa**: Calcular métricas de rendimiento (MAE, RMSE, R², MAPE)
4. **[20%] Interpretabilidad**: Usar SHAP y LIME para explicación del modelo
5. **[20%] Valor de Negocio**: Calcular ROI y punto de equilibrio
6. **[10%] Insights**: Proveer recomendaciones accionables

---

## Contexto de Negocio

### Startup HabitAlpes

HabitAlpes es una startup colombiana que ofrece servicios inmobiliarios incluyendo:
- Valoración y avalúo de propiedades
- Compra, remodelación y reventa
- Consultoría de contratos de arrendamiento
- Correcciones catastrales
- Informes de estratificación de barrios

### Propuesta de Valor

**Proceso Actual:**
- Tiempo de experto: 6 horas por valoración
- Costo: $9,500/hora × 6h = **$57,000 por apartamento**
- Capacidad: Hasta 500 apartamentos/mes

**Con Modelo ML:**
- Tiempo de experto: 1 hora por valoración
- Costo: $9,500/hora × 1h = **$9,500 por apartamento**
- **Ahorro: $47,500 por estimación precisa**

**Riesgo:**
- Subestimaciones >20M COP activan valoraciones manuales presenciales (costo adicional)
- Sobreestimaciones no son reportadas por los clientes

---

## Dataset

### Fuente
- **Archivo**: `data/apartamentos.csv`
- **Registros**: 43,013 apartamentos
- **Características**: 46 columnas
- **Período**: Últimos 2 meses del mercado inmobiliario de Bogotá

### Características Principales

**Variable Objetivo:**
- `precio_venta`: Precio de venta (COP)

**Características Físicas:**
- `area`: Metros cuadrados
- `habitaciones`: Número de habitaciones
- `banos`: Número de baños
- `parqueaderos`: Espacios de parqueo
- `piso`: Número de piso

**Ubicación:**
- `localidad`: Localidad
- `barrio`: Barrio
- `latitud`, `longitud`: Coordenadas geográficas
- `estrato`: Estrato socioeconómico (1-6)

**Amenidades:**
- Características binarias: `piscina`, `gimnasio`, `ascensor`, `vigilancia`, etc.

**Proximidad:**
- Distancia a estaciones de transporte masivo
- Distancia a parques

**Diccionario de datos completo**: `data/Diccionario de datos - apartamentos.html`

---

## Instalación

### Prerequisitos
- Python 3.9+
- pip

### Configuración

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd Taller_2_DataScience
```

2. **Crear un entorno virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
Taller_2_DataScience/
├── data/
│   ├── apartamentos.csv                 # Dataset original
│   ├── Diccionario de datos - apartamentos.html
│   ├── processed/                       # Divisiones train/test/validation
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│   └── results/                         # Archivos de salida
│
├── src/                                 # Scripts de Python (ejecutar en orden)
│   ├── utils.py                         # Funciones auxiliares
│   ├── 01_analisis_exploratorio.py      # Análisis Exploratorio de Datos
│   ├── 02_preprocesamiento.py           # Limpieza y división de datos
│   ├── 03_ingenieria_caracteristicas.py # Creación de características
│   ├── 04_modelado.py                   # Entrenamiento de modelos
│   ├── 05_evaluacion.py                 # Métricas cuantitativas
│   ├── 06_interpretabilidad.py          # Análisis SHAP & LIME
│   └── 07_valor_negocio.py              # ROI y punto de equilibrio
│
├── notebooks/                           # Notebooks de Jupyter
│   ├── 01_EDA_y_Preparacion.ipynb
│   ├── 02_Modelado_y_Evaluacion.ipynb
│   ├── 03_Analisis_Interpretabilidad.ipynb
│   └── 04_Valor_Negocio_e_Insights.ipynb
│
├── models/                              # Modelos entrenados (*.pkl)
│
├── reports/                             # Reportes y visualizaciones
│   ├── figures/                         # Gráficos generados
│   └── reporte_ejecutivo.md             # Reporte final
│
├── docs/
│   └── Taller2.pdf                      # Descripción del taller
│
├── requirements.txt                     # Dependencias de Python
├── .gitignore
└── README.md                            # Este archivo
```

---

## Instrucciones de Ejecución

### Opción 1: Ejecutar Scripts de Python (Recomendado para reproducibilidad)

Ejecutar scripts **secuencialmente** en el directorio `src/`:

```bash
# 1. Análisis Exploratorio de Datos
python src/01_analisis_exploratorio.py

# 2. Preprocesamiento y División de Datos
python src/02_preprocesamiento.py

# 3. Ingeniería de Características
python src/03_ingenieria_caracteristicas.py

# 4. Entrenamiento de Modelos (múltiples modelos)
python src/04_modelado.py

# 5. Evaluación de Modelos (métricas)
python src/05_evaluacion.py

# 6. Análisis de Interpretabilidad (SHAP & LIME)
python src/06_interpretabilidad.py

# 7. Cálculo de Valor de Negocio (ROI)
python src/07_valor_negocio.py
```

**Salidas Esperadas:**
- Datasets procesados en `data/processed/`
- Modelos entrenados en `models/`
- Visualizaciones en `reports/figures/`
- Métricas y reportes en `data/results/`

---

### Opción 2: Ejecutar Notebooks de Jupyter (Recomendado para exploración)

Iniciar Jupyter:
```bash
jupyter notebook
```

Ejecutar notebooks **en orden**:

1. **01_EDA_y_Preparacion.ipynb**
   - Cargar y explorar datos
   - Visualizar distribuciones y correlaciones
   - Identificar valores faltantes y valores atípicos

2. **02_Modelado_y_Evaluacion.ipynb**
   - Preprocesar datos
   - Entrenar múltiples modelos ML (Regresión Lineal, Random Forest, XGBoost, LightGBM)
   - Comparar métricas de rendimiento
   - Seleccionar el mejor modelo

3. **03_Analisis_Interpretabilidad.ipynb**
   - Importancia global de características con SHAP
   - Explicaciones individuales con SHAP
   - Interpretabilidad local con LIME
   - Insights del comportamiento del modelo

4. **04_Valor_Negocio_e_Insights.ipynb**
   - Calcular ahorros de costos
   - Análisis de ROI
   - Punto de equilibrio
   - Resumen ejecutivo y recomendaciones

**Nota:** Cada notebook puede ejecutarse independientemente ya que cargan salidas intermedias de los scripts.

---

## Resumen de Resultados

### Rendimiento del Modelo

| Modelo | R² Score | MAE (COP) | RMSE (COP) | MAPE (%) |
|-------|----------|-----------|------------|----------|
| Regresión Lineal | Por Determinar | Por Determinar | Por Determinar | Por Determinar |
| Random Forest | Por Determinar | Por Determinar | Por Determinar | Por Determinar |
| XGBoost | Por Determinar | Por Determinar | Por Determinar | Por Determinar |
| LightGBM | Por Determinar | Por Determinar | Por Determinar | Por Determinar |

**Modelo Seleccionado:** Por Determinar

---

### Valor de Negocio

**Ahorro por Estimación:**
- Sin ML: $57,000
- Con ML: $9,500
- **Ahorro Teórico: $47,500**

**Ahorro Real (considerando costos de errores):**
- Por Determinar (después de calcular costos de error)

**ROI:**
- Costo de Desarrollo: Por Determinar
- Ahorro Mensual: Por Determinar
- **Punto de Equilibrio: Por Determinar meses**

---

### Insights Clave

1. Por Determinar - Hallazgos de importancia de características
2. Por Determinar - Patrones geográficos
3. Por Determinar - Recomendación para HabitAlpes

Insights completos disponibles en: `reports/reporte_ejecutivo.md`

---

## Autores

- **Nombre Estudiante 1** - [GitHub](https://github.com/username1)
- **Nombre Estudiante 2** - [GitHub](https://github.com/username2)

**Curso:** MINE-4101 - Ciencia de Datos Aplicada
**Institución:** Universidad de los Andes
**Año:** 2025

---

## Licencia

Este proyecto es parte de una asignación académica para la Universidad de los Andes.

---

## Agradecimientos

- **HabitAlpes** (startup ficticia) por el caso de negocio
- Dataset obtenido de listados inmobiliarios de Bogotá
- Universidad de los Andes por la orientación del proyecto
