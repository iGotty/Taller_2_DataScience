# Guía de Ejecución
## HabitAlpes - Predicción de Precios de Apartamentos

Esta guía proporciona instrucciones paso a paso para ejecutar todos los componentes del proyecto.

---

## Prerequisitos

- Python 3.9 o superior
- Git
- 8GB+ RAM recomendado (para entrenamiento de modelos)
- ~2GB de espacio libre en disco

---

## Inicio Rápido

### 1. Clonar Repositorio

```bash
git clone <repository-url>
cd Taller_2_DataScience
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota**: La instalación puede tomar 5-10 minutos dependiendo de tu conexión a internet.

---

## Opción 1: Ejecutar Scripts de Python (Recomendado)

Ejecutar scripts **secuencialmente** en el directorio `src/`:

### Paso 1: Análisis Exploratorio de Datos

```bash
python src/01_analisis_exploratorio.py
```

**Salida esperada**:
- Figuras en `reports/figures/` (01_*.png hasta 14_*.png)
- CSV resumen en `data/results/resumen_eda.csv`
- **Duración**: ~2-3 minutos

### Paso 2: Preprocesamiento de Datos

```bash
python src/02_preprocesamiento.py
```

**Salida esperada**:
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/validation.csv`
- **Duración**: ~1-2 minutos

### Paso 3: Ingeniería de Características

```bash
python src/03_ingenieria_caracteristicas.py
```

**Salida esperada**:
- `data/processed/train_fe.csv`
- `data/processed/test_fe.csv`
- `data/processed/validation_fe.csv`
- **Duración**: ~1 minuto

### Paso 4: Entrenamiento de Modelos

```bash
python src/04_modelado.py
```

**Salida esperada**:
- Modelos entrenados en `models/` (archivos *.pkl)
- `data/results/comparacion_modelos.csv`
- `models/mejor_modelo.txt`
- **Duración**: ~10-15 minutos (varía según hardware)

**Nota**: Este es el script de mayor duración debido al ajuste de hiperparámetros.

### Paso 5: Evaluación de Modelos

```bash
python src/05_evaluacion.py
```

**Salida esperada**:
- Figuras en `reports/figures/` (15_*.png hasta 17_*.png)
- `data/results/metricas_validacion.csv`
- `data/results/resumen_evaluacion.txt`
- **Duración**: ~1 minuto

### Paso 6: Análisis de Interpretabilidad

```bash
python src/06_interpretabilidad.py
```

**Salida esperada**:
- Figuras en `reports/figures/` (18_*.png hasta 24_*.png)
- `data/results/importancia_caracteristicas_shap.csv`
- `data/results/resumen_interpretabilidad.txt`
- **Duración**: ~5-8 minutos

### Paso 7: Análisis de Valor de Negocio

```bash
python src/07_valor_negocio.py
```

**Salida esperada**:
- Figuras en `reports/figures/` (25_*.png hasta 28_*.png)
- `data/results/reporte_valor_negocio.txt`
- `data/results/metricas_valor_negocio.csv`
- **Duración**: ~30 segundos

### Tiempo Total de Ejecución

**~20-30 minutos** para todos los scripts (principalmente entrenamiento de modelos)

---

## Opción 2: Ejecutar Notebooks de Jupyter (Interactivo)

### 1. Iniciar Jupyter

```bash
jupyter notebook
```

Tu navegador se abrirá con la interfaz de Jupyter.

### 2. Navegar a Notebooks

Ir al directorio `notebooks/`.

### 3. Ejecutar en Orden

**Importante**: Ejecutar notebooks en este orden exacto:

1. **01_EDA_y_Preparacion.ipynb**
   - Explora el dataset
   - Genera visualizaciones
   - Analiza valores faltantes y correlaciones

2. **02_Modelado_y_Evaluacion.ipynb**
   - Preprocesa datos
   - Ingeniería de características
   - Entrena múltiples modelos
   - Evalúa rendimiento

3. **03_Analisis_Interpretabilidad.ipynb**
   - Ejecuta análisis SHAP
   - Genera explicaciones LIME
   - Interpreta comportamiento del modelo

4. **04_Valor_Negocio_e_Insights.ipynb**
   - Calcula ROI y punto de equilibrio
   - Proporciona recomendaciones de negocio
   - Genera resumen ejecutivo

### 4. Consejos de Ejecución

- **Ejecutar celdas secuencialmente** (Shift + Enter)
- **Descomentar** líneas con `# %run ../src/XX_script.py` para ejecutar scripts subyacentes
- Algunas celdas pueden tomar varios minutos (entrenamiento de modelos, cálculo SHAP)
- Asegurar que cada notebook se complete antes de pasar al siguiente

---

## Lista de Verificación

Después de ejecutar todos los scripts/notebooks, verificar:

### Archivos Generados

```
Taller_2_DataScience/
├── data/
│   ├── processed/
│   │   ├── train.csv ✓
│   │   ├── test.csv ✓
│   │   ├── validation.csv ✓
│   │   ├── train_fe.csv ✓
│   │   ├── test_fe.csv ✓
│   │   └── validation_fe.csv ✓
│   └── results/
│       ├── resumen_eda.csv ✓
│       ├── comparacion_modelos.csv ✓
│       ├── metricas_validacion.csv ✓
│       ├── importancia_caracteristicas_shap.csv ✓
│       ├── metricas_valor_negocio.csv ✓
│       └── archivos *.txt ✓
├── models/
│   ├── *.pkl (archivos de modelos) ✓
│   ├── scaler.pkl ✓
│   └── mejor_modelo.txt ✓
└── reports/
    └── figures/
        └── 01_*.png hasta 28_*.png ✓
```

### Cantidad Esperada de Figuras

- **28 archivos PNG** en `reports/figures/`
- Cubriendo: EDA, evaluación de modelos, SHAP, LIME, valor de negocio

---

## Solución de Problemas

### Problema: Módulo No Encontrado

```
ModuleNotFoundError: No module named 'xxx'
```

**Solución**:
```bash
# Asegurar que el entorno virtual esté activado
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstalar requirements
pip install -r requirements.txt
```

### Problema: Error de Memoria

```
MemoryError: Unable to allocate array
```

**Solución**:
- Reducir tamaño de muestra en scripts de interpretabilidad
- Cerrar otras aplicaciones
- Usar una máquina con más RAM (8GB+ recomendado)

### Problema: Cálculo SHAP Lento

**Esperado**: El análisis SHAP puede tomar 5-10 minutos

**Solución**:
- Esto es normal para datasets de este tamaño
- El script muestrea 1,000 instancias para acelerar el cálculo
- Ser paciente o reducir tamaño de muestra en `src/06_interpretabilidad.py` (línea ~XX)

### Problema: Archivo No Encontrado

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/train.csv'
```

**Solución**:
- Ejecutar scripts en orden (preprocesamiento antes de modelado)
- Verificar que los scripts previos se completaron exitosamente
- Verificar que el directorio de trabajo es la raíz del proyecto

---

## Benchmarks de Rendimiento

Probado en:
- **CPU**: Intel i7 / AMD Ryzen 5 equivalente
- **RAM**: 16GB
- **SO**: Ubuntu 22.04 / macOS / Windows 11

| Script | Duración |
|--------|----------|
| 01_analisis_exploratorio.py | 2-3 min |
| 02_preprocesamiento.py | 1-2 min |
| 03_ingenieria_caracteristicas.py | 1 min |
| 04_modelado.py | 10-15 min |
| 05_evaluacion.py | 1 min |
| 06_interpretabilidad.py | 5-8 min |
| 07_valor_negocio.py | <1 min |
| **Total** | **20-30 min** |

---

## Ejemplos de Salidas

### Muestra de Salida en Consola

```
================================================================================
  ANÁLISIS EXPLORATORIO DE DATOS - PROYECTO HABITALPIES
================================================================================

--------------------------------------------------------------------------------
  1. CARGANDO DATOS
--------------------------------------------------------------------------------
Cargando datos desde .../data/apartamentos.csv...
Cargados 43,013 registros con 46 columnas

--------------------------------------------------------------------------------
  2. DIMENSIONES DEL DATASET
--------------------------------------------------------------------------------
Número de filas: 43,013
Número de columnas: 46
...
```

### Muestra de Figuras Generadas

```
reports/figures/
├── 01_valores_faltantes.png          # Visualización de datos faltantes
├── 02_distribucion_precio_venta.png  # Distribución variable objetivo
├── 15_actual_vs_predicho.png         # Predicciones del modelo
├── 18_resumen_shap.png                # Importancia de características
└── 26_analisis_punto_equilibrio.png   # Análisis ROI
```

---

## Notas Adicionales

### El Orden de Ejecución es Crítico

Los scripts tienen dependencias:
1. EDA (independiente)
2. Preprocesamiento (requiere datos crudos)
3. Ingeniería de Características (requiere datos preprocesados)
4. Modelado (requiere datos con FE)
5. Evaluación (requiere modelo entrenado)
6. Interpretabilidad (requiere modelo entrenado)
7. Valor de Negocio (requiere métricas de evaluación)

**¡No omitir pasos ni ejecutar fuera de orden!**

### Reproducibilidad

- Todos los scripts usan `random_state=42` para reproducibilidad
- Los resultados deben ser idénticos entre ejecuciones (excepto diferencias menores de punto flotante)

### Uso de Recursos

- **Disco**: ~500MB para modelos y datos procesados
- **Memoria**: Pico de ~6GB durante entrenamiento de modelos
- **CPU**: Utiliza todos los núcleos para entrenamiento de modelos

---

## Obtener Ayuda

Si encuentras problemas:

1. **Consultar esta guía**: Los problemas más comunes están cubiertos
2. **Leer mensajes de error**: Usualmente indican el problema
3. **Verificar requirements**: Asegurar que todas las dependencias están instaladas
4. **Verificar datos**: Asegurar que `data/apartamentos.csv` existe y no está corrupto

---

## Referencia de Estructura del Proyecto

```
Taller_2_DataScience/
├── data/                  # Archivos de datos
│   ├── apartamentos.csv   # Dataset crudo (43K registros)
│   ├── processed/         # Divisiones train/test/val
│   └── results/           # Salidas CSV y reportes
│
├── src/                   # Scripts de Python (ejecutar en orden 01-07)
│   ├── utils.py
│   ├── 01_analisis_exploratorio.py
│   ├── 02_preprocesamiento.py
│   ├── 03_ingenieria_caracteristicas.py
│   ├── 04_modelado.py
│   ├── 05_evaluacion.py
│   ├── 06_interpretabilidad.py
│   └── 07_valor_negocio.py
│
├── notebooks/             # Notebooks de Jupyter (interactivos)
│   ├── 01_EDA_y_Preparacion.ipynb
│   ├── 02_Modelado_y_Evaluacion.ipynb
│   ├── 03_Analisis_Interpretabilidad.ipynb
│   └── 04_Valor_Negocio_e_Insights.ipynb
│
├── models/                # Modelos entrenados (generados)
├── reports/               # Salidas y visualizaciones
├── requirements.txt       # Dependencias de Python
├── README.md              # Descripción del proyecto
└── EXECUTION_GUIDE.md     # Este archivo
```

---

**Última Actualización**: 2025
**Mantenido por**: Equipo de Ciencia de Datos
