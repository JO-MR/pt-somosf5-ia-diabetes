# Explorador de Sesgos: Mini Aplicación Didáctica de Machine Learning  
### Prueba Técnica – Formador/a en IA | Fundación Somos F5
## Accesos rápidos, al script/notebook creado google colab y al despliegue de la aplicación de streamlit en Hugging Face

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U3XrIRgcRPddmpmYfP5FWmLsm565ZcVb)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Space-blue)](https://huggingface.co/spaces/JonasDMR/diabetes-streamlit-jonas)



Este repositorio contiene una **actividad educativa completa** para enseñar Machine Learning, análisis ético, visualización de datos e introducción a IA generativa.  

Combina:

- Modelos clásicos de ML  
- Análisis de sesgos  
- Métricas y visualizaciones pedagógicas  
- Explicabilidad accesible con un mock de LLM  
- Una mini interfaz Streamlit desplegada en la nube  

Diseñado para cumplir **todas las exigencias técnicas y pedagógicas** de la Prueba Técnica de Somos F5.

---

## Índice

1. Descripción general del proyecto  
2. Qué hace el código, paso a paso  
3. Qué aprendería el alumnado al ejecutar este ejemplo  
4. Cómo usar este recurso en una clase  
5. Qué podría salir mal y cómo solucionarlo  
6. Ideas para adaptar la actividad a distintos niveles  
7. Comparación de hiperparámetros del Random Forest  
8. Mini interfaz con Streamlit desplegada en Hugging Face
9. Conclusión

---
##  Accesos rápidos mediante QR
Con el objetivo de faciliar la accesibilidad, en diferentes dispositivos, he creado tres QR, que permiten visualizar la interfaz de Streamlit en Hugging Face, el código en Google Colab y el repositorio de la Prueba Técnica, almacenado en Github.

![QRs del proyecto](https://raw.githubusercontent.com/JO-MR/pt-somosf5-ia-diabetes/main/QRs.png)

---

#  Estructura del repositorio

```text
.
├── PT_diabetes.ipynb         # Notebook principal: EDA, modelos, métricas, sesgo y mock LLM
├── resultados_modelos.csv     # Métricas exportadas para la mini aplicación Streamlit
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Documento educativo (este archivo)
└── src/
    └── app.py                 # Mini interfaz Streamlit en Hugging Face
```

##  1. Descripción general del proyecto

La mini aplicación **Explorador de Sesgos** permite al alumnado:

- Cargar y explorar un dataset real.  
- Preprocesar datos con técnicas profesionales.  
- Entrenar y comparar modelos de Machine Learning.  
- Interpretar métricas y visualizaciones.  
- Detectar posibles sesgos (por ejemplo, por edad).  
- Generar explicaciones accesibles mediante un mock de IA generativa.  
- Interactuar con una mini interfaz web sin necesidad de programar.

Este enfoque integra **solidez técnica**, **pedagogía inclusiva** y **una mirada ética a la IA**.

---
## 2. Qué hace el código, paso a paso

El notebook `PT_diabetes.ipynb` desarrolla una experiencia formativa completa, donde cada bloque combina **técnica**, **pedagogía**, **pensamiento crítico** y **ética aplicada**.  
A continuación se presenta una descripción clara y estructurada de cada parte del flujo.

---

###  Bloque 1 — Carga del dataset (CSV público) y descripción inicial
### Fuente del dataset

Origen: repositorio público de Plotly (datasets educativos).

Pima Indians Diabetes Dataset desde URL pública.

Formato: CSV.

URL directa: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

Licencia: uso abierto para fines educativos.

En este bloque se realiza:

- Carga del dataset público en formato CSV.
- Revisión de tamaño, columnas, tipos de variables y variable objetivo.
- Descripción de la fuente y licencia del dataset.
- Primer análisis reflexivo sobre la calidad y limitaciones de los datos.

  ## 📂 Dataset utilizado

- **Dataset original:**  
  [Pima Indians Diabetes Dataset (CSV)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

- **Versión utilizada en el proyecto (preprocesada):**  
  [`dataset_diabetes.csv`](./dataset_diabetes.csv)


**Propósito pedagógico:**  
Introducir al alumnado a trabajar con datos reales, validar su estructura y comprender el contexto antes de modelar.

---

###  Bloque 2 — Exploración inicial del dataset

Incluye:

- Visualización de las primeras filas.
- `info()`, `describe()`, revisión de nulos y tipos.
- Detección de posibles problemas estructurales (valores anómalos, distribución irregular del target).

**Propósito pedagógico:**  
Aprender a “leer” un dataset y desarrollar habilidades críticas para evaluar si los datos son aptos para un modelo de IA.

---

###  Bloque 3 — Análisis Exploratorio de Datos (EDA)
---

### Visualizaciones incluidas en el notebook

El notebook `PT_diabetes.ipynb` incorpora todas las visualizaciones necesarias para comprender los datos y el comportamiento de los modelos.  
Estas visualizaciones se encuentran generadas en los Bloques 2, 3 y 5 del Colab e incluyen:

- Histogramas de variables clave  
- Boxplots para detección de valores atípicos  
- Matriz de correlación  
- Curva ROC de los modelos  
- Matriz de confusión  
- Gráficos comparativos de métricas
- Correlaciones entre variables.
- Análisis visual de patrones y posibles sesgos

Estas figuras permiten al alumnado **interpretar los resultados de manera visual**, fomentando pensamiento crítico y análisis basado en evidencia.

Puedes revisar las visualizaciones directamente aquí:  
**Notebook en Google Colab:** https://colab.research.google.com/drive/1U3XrIRgcRPddmpmYfP5FWmLsm565ZcVb

---

**Propósito pedagógico:**  
El alumnado aprende a interpretar gráficos, identificar patrones y formular hipótesis basadas en evidencia visual.

---

###  Bloque 4 — Preprocesamiento y preparación del dataset

Este bloque implementa:

- Reemplazo de valores imposibles por `NaN`.
- Imputación mediante mediana.
- Escalado de características numéricas.
- División train/test con `stratify` y `random_state=42`.

**Propósito pedagógico:**  
Comprender por qué el preprocesamiento influye en el rendimiento del modelo y cómo puede introducir o mitigar sesgos.

---

###  Bloque 5 — Entrenamiento y comparación de modelos de Machine Learning

Incluye dos modelos clásicos:

- **Regresión Logística**
- **Random Forest**

Se comparan mediante:

- Accuracy  
- Recall (clase positiva)  
- F1-macro  
- Matriz de confusión  
- Curva ROC individual y combinada  

**Propósito pedagógico:**  
Aprender qué aprende cada modelo, cómo interpretar resultados y cómo justificarlos en contextos educativos y reales.

---

###  Bloque 6 — Sesgos, explicabilidad con IA generativa y reflexión ética

Este bloque integra:

- Cálculo de métricas por grupos de edad.
- Visualización de desigualdades (p. ej., menor recall en personas jóvenes).
- Identificación de riesgos de sesgo y discusión sobre su impacto.
- Implementación de un **mock de IA generativa** para explicar resultados en lenguaje accesible.

**Propósito pedagógico:**  
Fomentar pensamiento crítico sobre el impacto social de la IA y enseñar a comunicar resultados de manera inclusiva.

## Ejemplo del prompt utilizado en el mock de IA generativa

El proyecto incluye una función que simula cómo un modelo de IA generativa explicaría los resultados de un modelo de Machine Learning, transformando métricas técnicas en un lenguaje claro y accesible.
El prompt generado, está encluido en el notebook de google colab dentro del bloque 6 - punto dos.

---
###  Bloque 7 — Comparación de hiperparámetros del Random Forest (opcional)

Incluye:

- Variación de hiperparámetros clave.
- Comparación estructurada de métricas.
- Selección fundamentada del mejor modelo.

**Propósito pedagógico:**  
Introducir a la optimización de modelos y al razonamiento basado en evidencia.

---

###  Bloque 8 — Mini interfaz con Streamlit desplegada en Hugging Face

Este bloque conecta el proyecto técnico con una experiencia interactiva:

- Lectura de `resultados_modelos.csv`.
- Interfaz para explorar métricas sin programar.
- Área interactiva para probar el mock LLM.
- Despliegue en Hugging Face Spaces.

 Acceso directo: https://huggingface.co/spaces/JonasDMR/diabetes-streamlit-jonas

**Propósito pedagógico:**  
Mostrar cómo un modelo de IA puede integrarse en una herramienta educativa real y accesible para todo tipo de alumnado.

---

## 3. Qué aprendería el alumnado, al ejecutar este ejemplo

Este proyecto está diseñado como una experiencia educativa completa.  
Cada bloque del notebook y cada componente del proyecto (modelos, métricas, análisis de sesgo, IA generativa y mini interfaz) contribuye a desarrollar **competencias técnicas, analíticas, éticas y comunicativas**, esenciales en el aprendizaje de la Inteligencia Artificial.

A continuación se detalla qué aprendería el alumnado de manera progresiva:

---

###  3.1. Competencias técnicas en Ciencia de Datos y Machine Learning

- Comprender el flujo completo de un proyecto de IA:  
  **EDA → Preprocesamiento → Modelado → Evaluación → Sesgos → Comunicación.**
- Manejar datasets reales en formato CSV y diagnosticar su calidad.
- Aplicar preprocesamiento profesional:  
  imputación, escalado, detección de valores anómalos, división estratificada.
- Entrenar modelos clásicos de ML:  
  **Regresión Logística** y **Random Forest**.
- Interpretar métricas clave:  
  Accuracy, Recall, F1-macro.
- Analizar curvas ROC y matrices de confusión.
- Comprender el impacto del hiperparámetro en Random Forest (Bloque 7).

---

###  3.2. Competencias analíticas y de interpretación

El estudiante desarrollará la capacidad de:

- Leer gráficos y detectar patrones relevantes.  
- Identificar desequilibrios y posibles sesgos dentro de los datos.  
- Comparar modelos no solo por su precisión, sino por su rendimiento en grupos específicos.  
- Analizar el rendimiento segmentado (por ejemplo, por grupos de edad).  
- Formular hipótesis basadas en evidencia.

---

###  3.3. Competencias éticas y de pensamiento crítico

Gracias al análisis de sesgos y al mock de IA generativa, el alumnado aprenderá a:

- Reflexionar sobre **quién se beneficia** y **quién puede verse perjudicado** por un modelo de IA.  
- Reconocer los riesgos de sesgos algorítmicos en contextos sensibles (salud, educación…).  
- Comprender la importancia del *recall* en problemas de salud (minimizar falsos negativos).  
- Diseñar explicaciones claras y transparentes para usuarios no técnicos.  
- Practicar un enfoque responsable hacia la Inteligencia Artificial.

---

###  3.4. Competencias comunicativas

El mock LLM incluido en el proyecto permite al alumnado practicar:

- Redacción de prompts estructurados y efectivos.  
- Transformación de métricas técnicas en explicaciones accesibles.  
- Comunicación ética y clara de los resultados de un modelo de IA.  
- Presentación de conclusiones comprensibles para equipos multidisciplinares.

---

###  3.5. Competencias digitales y de integración tecnológica

Gracias a la mini interfaz Streamlit en Hugging Face, el alumnado experimenta:

- Cómo conectar un pipeline de ML con una aplicación interactiva.  
- Cómo explorar modelos sin necesidad de programar.  
- Cómo una interfaz puede mejorar la accesibilidad de la IA.  
- Cómo desplegar aplicaciones simples en la nube.

---

###  3.6. Competencias aplicadas al contexto formativo real

El estudiante entenderá:

- Cómo estructurar un proyecto para enseñarlo a otras personas.  
- Cómo documentar procesos de IA con claridad pedagógica.  
- Cómo facilitar actividades prácticas y debates éticos en clase.  
- Cómo convertir un ejercicio de ML en un recurso didáctico transformador.

---

###  Conclusión del Punto 3

Al completar esta práctica, el alumnado no solo aprende a entrenar modelos de ML, sino que desarrolla:

- **Pensamiento crítico**,  
- **Conciencia ética**,  
- **Capacidad de comunicación**,  
- **Resolución de problemas**,  
- **Razonamiento basado en datos**,  
- **Comprensión de sesgos algorítmicos**,  
- **Competencias para el mundo laboral de la IA responsable**.

---

## 4. Cómo se podría usar este recurso en una clase (dinámica, tiempos y actividades)

Este proyecto está diseñado para funcionar como una **actividad formativa completa** dentro de un curso de Ciencia de Datos, IA Responsable, Análisis de Datos o Programación con Python.  
La propuesta metodológica combina explicación, práctica guiada, reflexión ética y uso de herramientas tecnológicas accesibles para todo tipo de alumnado.

A continuación se describe una **propuesta de sesión formativa de 90–120 minutos** basada en aprendizaje activo y trabajo colaborativo.

---

###  4.1. Estructura temporal recomendada

| Fase | Duración | Objetivo |
|------|----------|----------|
| **1. Exploración inicial del dataset** | 15–20 min | Comprender los datos y activar conocimientos previos |
| **2. Preprocesamiento y modelado** | 25–30 min | Preparar los datos y entrenar modelos clásicos |
| **3. Evaluación y análisis de sesgos** | 20–30 min | Interpretar métricas, visualizar resultados y detectar sesgos |
| **4. Mini aplicación Streamlit + IA generativa** | 20–30 min | Explorar modelos desde una interfaz y generar explicaciones accesibles |
| **5. Debate ético y cierre** | 10 min | Reflexionar sobre el impacto real y el uso responsable de la IA |

Duración total: **90–120 min**

---

###  4.2. Dinámica recomendada de la clase

La actividad está pensada para trabajarse en:

- **Parejas** → para favorecer razonamiento conjunto.  
- **Pequeños grupos** → para discutir interpretaciones y decisiones.  
- **Puesta en común** → para consensuar aprendizajes y reflexiones éticas.

El/la formador/a actúa como guía, acompañando el proceso, realizando preguntas clave, promoviendo pensamiento crítico, desde el aprender haciendo.

---

###  4.3. Actividad práctica por fases

---

### **Fase 1 — Exploración del dataset (15–20 min)**  
**Acciones:**

- Ejecutar los bloques iniciales del notebook.  
- Observar distribuciones e identificar posibles problemas de calidad.  
- Preguntas orientadoras del formador/a:
  - *¿Qué variables podrían tener impacto en la diabetes?*
  - *¿Qué problemas detectáis en los datos?*
  - *¿Qué decisiones deberíamos tomar antes de modelar?*

**Aprendizaje esperado:**  
Comprender la importancia del análisis previo antes de entrenar modelos.

---

### **Fase 2 — Preprocesamiento y modelado (25–30 min)**  
**Acciones:**

- Completar el preprocesado con imputación, escalado y partición.
- Entrenar los dos modelos: Regresión Logística y Random Forest.
- Comparar su rendimiento en test.

**Aprendizaje esperado:**  
Entender cómo las decisiones técnicas afectan al comportamiento del modelo.

---

### **Fase 3 — Evaluación y análisis de sesgos (20–30 min)**  
**Acciones:**

- Interpretar accuracy, recall y F1-macro.  
- Revisar matriz de confusión y curva ROC.  
- Analizar métricas por grupos de edad para detectar sesgo.

**Preguntas clave:**

- *¿Qué grupo está peor representado o peor detectado?*  
- *¿Qué implicaciones tiene en un contexto sanitario?*  
- *¿Cómo podríamos mitigar este sesgo?*

**Aprendizaje esperado:**  
Tomar conciencia de que un modelo puede ser bueno globalmente, pero injusto localmente.

---

### **Fase 4 — Mini interfaz Streamlit + IA generativa (20–30 min)**  
**Acciones:**

- Abrir la app en Hugging Face.  
- Probar diferentes modelos y métricas.  
- Introducir un riesgo de sesgo y observar la explicación generada por el mock LLM.  

 App: https://huggingface.co/spaces/JonasDMR/diabetes-streamlit-jonas

**Aprendizaje esperado:**  
Experimentar cómo una interfaz accesible, mejora la comprensión y comunicación de resultados.

---

### **Fase 5 — Debate ético y cierre (10 min)**  
**Acciones:**

Discusión abierta sobre:

- Impacto del sesgo en salud.  
- Importancia del uso responsable de la IA.  
- Limitaciones de los modelos de ML y de los LLM.  
- Propuestas del alumnado para mejorar el sistema.

**Aprendizaje esperado:**  
Desarrollar competencias críticas para evaluar sistemas de IA desde una perspectiva social y ética.

---

###  4.4. Objetivo educativo final

El propósito de esta práctica es que el alumnado:

- Comprenda el proceso completo de creación de un modelo de IA.  
- Desarrolle criterio técnico y ético.  
- Sea capaz de explicar resultados de forma accesible.  
- Reflexione sobre posibles injusticias provocadas por modelos automáticos.  
- Aprenda a integrar IA en herramientas reales (mini apps, dashboards, etc.).

El proyecto no solo enseña **técnica**, sino **cómo enseñar IA de forma inclusiva, clara y responsable**.

---

## 5. Qué podría salir mal y cómo solucionarlo

Este proyecto está pensado para ser robusto, reproducible y fácil de ejecutar, pero como ocurre en cualquier entorno educativo, pueden surgir errores.  
A continuación se presenta una tabla completa con **los problemas más comunes**, su **causa probable** y **cómo resolverlos**, con el objetivo de facilitar el aprendizaje autónomo del alumnado y anticipar dificultades en el aula.

---

###  Problemas comunes durante la ejecución del notebook

| Problema | Causa probable | Solución recomendada |
|----------|----------------|-----------------------|
| **El CSV no carga o da error de ruta** | Archivo fuera del directorio esperado | Verificar que el CSV esté en la carpeta correcta o usar rutas relativas |
| **El dataset aparece con muchos ceros imposibles** | Errores de medición en el dataset original | Reemplazar ceros por `NaN` y aplicar imputación como se hace en el notebook |
| **Persisten NaN después del preprocesamiento** | Faltó imputación o columnas nuevas heredaron NaN | Revisar imputación, ejecutar celdas en orden o revisar columnas añadidas |
| **El modelo Random Forest da resultados incoherentes** | División train/test no estratificada | Asegurar `stratify=y` en `train_test_split` |
| **Resultados distintos a los del formador** | Semilla aleatoria distinta | Confirmar que todas las operaciones usan `random_state=42` |
| **Las curvas ROC no se dibujan** | El modelo no implementa `predict_proba` | Usar modelos que sí generen probabilidades (como LogReg y RandomForest) |
| **Warnings molestos en sklearn** | Versiones diferentes de librerías | Instalar versiones actualizadas (`pip install -U scikit-learn`) |

---

###  Problemas relacionados con entrenamiento y métricas

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| **Accuracy alto, recall muy bajo** | Dataset desbalanceado o threshold inadecuado | Focalizarse en *recall*, revisar distribución, considerar rebalanceo |
| **Overfitting evidente en Random Forest** | Árboles demasiado profundos | Ajustar `max_depth`, revisar sección de hiperparámetros |
| **GridSearch muy lento** | Espacio de búsqueda grande | Reducir combinaciones o usar RandomizedSearchCV |
| **F1-macro muy bajo en jóvenes** | Sesgo del dataset | Analizar por grupos, discutir mitigación (más datos, reponderación, etc.) |

---

###  Problemas al ejecutar la mini interfaz Streamlit (Hugging Face)

| Problema | Causa | Solución |
|----------|--------|----------|
| **Error “FileNotFoundError: resultados_modelos.csv”** | El archivo no está en `/src/` | Subir el CSV al directorio `src/` dentro del Space |
| **La app no arranca y se queda en “Building…”** | Dependencias faltantes | Confirmar que `requirements.txt` incluye: `streamlit`, `pandas`, `scikit-learn` |
| **Pantalla en blanco al cargar la app** | Error en `app.py` | Revisar logs en Hugging Face → corregir ruta, variable o indentación |
| **El mock LLM no genera texto** | Inputs vacíos | Validar que los campos del formulario estén completos |
| **El Space falla tras actualizar archivos** | Caché desactualizada | Reiniciar Space: en Hugging Face → Settings → Restart |

---

###  Problemas conceptuales que pueden surgir en clase

| Duda o dificultad | Qué significa | Cómo acompañar al alumnado |
|-------------------|----------------|----------------------------|
| *“¿Por qué accuracy no basta?”* | No refleja bien casos desbalanceados | Mostrar ejemplos con falsos negativos en salud |
| *“¿Por qué el modelo falla más con jóvenes?”* | Sesgo por subrepresentación | Explicar importancia de diversidad en datos |
| *“¿Por qué imputamos valores faltantes?”* | Datos reales nunca están limpios | Explicar riesgos de modelos clínicos sin imputación |
| *“¿Por qué Random Forest funciona mejor?”* | Modelo no lineal que captura más patrones | Comparar gráficas y métricas lado a lado |
| *“¿Qué es un LLM mock?”* | Simulación de IA generativa para fines pedagógicos | Destacar cómo comunicar IA a público no técnico |

---

###  Problemas adicionales en un entorno formativo real

- Estudiantes que no ejecutan las celdas en orden →  
  **Solución:** reiniciar entorno y ejecutar del principio al final.  

- Ordenadores lentos →  
  **Solución:** usar Google Colab (todo corre en la nube).  

- Dudas éticas profundas →  
  **Solución:** dedicar 10 minutos al cierre para reflexiones sobre IA responsable.

---

###  Conclusión del Punto 5

Este apartado permite anticipar errores y convertir cada dificultad en una oportunidad de aprendizaje.  
El objetivo es que el alumnado:

- comprenda que fallar es parte del proceso,  
- sepa diagnosticar problemas,  
- desarrolle autonomía técnica,  
- y aprenda a pensar de forma rigurosa y ética sobre la IA.  

Este apoyo estructurado facilita que el formador/a atienda mejor a grupos diversos y fomente un aprendizaje seguro y accesible.

---

## 6. Ideas para adaptar la actividad a distintos niveles  
### Enfoque inclusivo, para grupos diversos y en situación de vulnerabilidad

El proyecto está diseñado para ser **accesible, escalable y adaptable** a distintos ritmos de aprendizaje, competencias previas y necesidades del alumnado.  
Dado que la misión de Somos F5 es ofrecer formación tecnológica a personas procedentes de entornos diversos y, en ocasiones, en situación de vulnerabilidad, este recurso se ha planteado desde:

- la **inclusión**,  
- el **acompañamiento gradual**,  
- la **reducción de barreras cognitivas**,  
- y la **facilitación de aprendizajes significativos**.

A continuación se detallan propuestas concretas para adaptar la actividad según niveles y contextos.

---

###  6.1. Nivel inicial  
**Pensado para alumnado sin experiencia previa en IA o programación.**  
El objetivo es **generar confianza**, reducir ansiedad tecnológica y asegurar comprensión conceptual.

**Adaptaciones recomendadas:**

- Entregar el notebook **ya pre-ejecutado**, con resultados visibles.  
- Pedir solo que ejecuten celdas, sin necesidad de modificar código.  
- Explicar las métricas con ejemplos cotidianos:
  - *Recall = “de todas las personas con diabetes, ¿a cuántas detecto?”*  
- Usar la mini app Streamlit como recurso principal para:
  - elegir modelo,  
  - ver métricas,  
  - generar explicación con el mock LLM.  
- Enfocar más en **qué significa** el resultado que en cómo se calcula.  
- Actividades prácticas:
  - interpretar la matriz de confusión con un caso narrado;  
  - reflexionar sobre por qué es importante detectar a tiempo enfermedades.  

**Objetivo pedagógico:**  
Acompañar al alumno a comprender la IA como **herramienta comprensible y útil**, no como algo inaccesible o intimidante.

---

###  6.2. Nivel intermedio  
**Pensado para alumnado que ya domina lo básico de Python y quiere profundizar.**

**Adaptaciones recomendadas:**

- Pedir al alumnado que realice ellos mismos:
  - la imputación de nulos,  
  - el escalado,  
  - el ajuste de hiperparámetros,  
  - alguna modificación visual (EDA adicional).  
- Comparar **qué cambia** al modificar parámetros del Random Forest.  
- Experimentar con umbrales de clasificación (`predict_proba`).  
- Introducir un pequeño reto:
  - *“Encuentra un sesgo no explorado en el notebook.”*  
- Usar el mock LLM para redactar explicaciones más elaboradas:
  - informes para pacientes,  
  - comunicaciones para equipos médicos,  
  - presentaciones para clase.  

**Objetivo pedagógico:**  
Promover autonomía, exploración y razonamiento propio.

---

###  6.3. Nivel avanzado  
**Pensado para alumnado con bases sólidas que busca retos mayores o preparar portfolio profesional.**

**Adaptaciones recomendadas:**

- Añadir técnicas de **explicabilidad avanzada**:
  - SHAP, LIME, Permutation Importance.  
- Implementar **fairness metrics** (Equal Opportunity, Demographic Parity).  
- Pedir al alumnado que:
  - reentrene modelos ajustando pesos por clase,  
  - genere versiones más robustas del modelo,  
  - documente sesgos y proponga mitigaciones reales.  
- Conectar con la mini interfaz Streamlit:
  - añadir sliders,  
  - mejorar gráficos,  
  - permitir cargar un dataset propio,  
  - desplegar una nueva app.  
- Introducir proyectos en grupo:
  - *“Diseña una solución IA Responsable para un caso social real.”*

**Objetivo pedagógico:**  
Desarrollar pensamiento crítico, capacidad de diseño de soluciones y preparación para entornos laborales reales.

---

###  6.4. Estrategias de acompañamiento inclusivo (clave para Somos F5)

En cualquier nivel, se recomiendan estas prácticas para garantizar que nadie quede atrás:

- **Lenguaje accesible:** evitar tecnicismos sin explicación.  
- **Aprendizaje visual:** reforzar conceptos mediante gráficos y analogías.  
- **Trabajo en parejas o grupos pequeños:** ideal para aprendizaje entre iguales.  
- **Puesta en común frecuente:** validar comprensión y construir confianza.  
- **Reconocer distintos ritmos de aprendizaje:** ofrecer alternativas si alguien se bloquea.  
- **Celebrar pequeños avances:** fundamental para aumentar autoestima en perfiles vulnerables.  
- **Aprender haciendo:** todo el proyecto fomenta manipulación práctica y experimentación.  
- **Vincular la IA con problemas reales cercanos:** salud, inclusión, oportunidades laborales.

---

###  6.5. Conclusión del Punto 6

Este proyecto no solo es técnicamente sólido:  
está diseñado para **empoderar** al alumnado, facilitar aprendizajes significativos y promover un enfoque **responsable, ético e inclusivo de la IA**.

Siguiendo estas adaptaciones, el recurso puede ser utilizado con éxito en grupos:

- diversos,  
- multidisciplinares,  
- con diferentes niveles de experiencia,  
- y en situación de vulnerabilidad.

La clave es acompañar, contextualizar y mostrar que la tecnología es una herramienta para **crear oportunidades**, no barreras.

---

## 7. Comparación de hiperparámetros del Random Forest (opcional, muy valorable)

Aunque la Prueba Técnica no exige obligatoriamente la optimización de modelos, explorar los hiperparámetros del Random Forest permite al alumnado comprender:

- cómo las decisiones técnicas afectan al rendimiento del modelo,  
- cómo se equilibra complejidad vs. generalización,  
- y cómo justificar mejoras basadas en evidencia.

Este bloque incluye dos enfoques: exploración manual y búsqueda sistemática.

---

###  7.1. Exploración manual de hiperparámetros

El notebook compara diferentes configuraciones modificando parámetros como:

- `n_estimators` (número de árboles)  
- `max_depth` (profundidad máxima)  
- `min_samples_split`  
- `criterion` (`gini` o `entropy`)  

Cada configuración se evalúa con las métricas ya trabajadas:

- Accuracy  
- Recall (clase positiva)  
- F1-macro  

**Propósito pedagógico:**  
El alumnado aprende que un modelo no es una “caja negra”, sino un sistema ajustable donde cada opción tiene implicaciones directas.

---

###  7.2. GridSearchCV — Búsqueda sistemática del mejor modelo

El notebook también implementa `GridSearchCV`, lo que permite:

- probar múltiples combinaciones de hiperparámetros,  
- usar validación cruzada,  
- optimizar según la métrica más adecuada (F1-macro),  
- obtener un Random Forest final más equilibrado y robusto.

**Parámetros explorados en GridSearch:**

- `n_estimators`  
- `max_depth`  
- `min_samples_split`  
- `max_features`  

**Métricas comparadas:**  
Las mismas que en el resto del proyecto, para garantizar coherencia y comprensión progresiva.

---

###  7.3. Interpretación formativa de los resultados

El análisis no se limita a elegir el modelo con mejor métrica:  
se explica **por qué** esa combinación funciona mejor.

Ejemplos de reflexiones que incorpora el notebook:

- Más árboles pueden aportar estabilidad, pero aumentan el coste computacional.  
- Profundidades muy grandes pueden causar overfitting.  
- Ciertos criterios (como `entropy`) captan mejor información en datasets con distribuciones específicas.  
- Un modelo con mayor F1-macro puede ser más adecuado que uno con mayor accuracy en contextos sensibles como salud.

**Aprendizaje esperado:**  
El alumnado desarrolla una mentalidad basada en *“interpretar evidencia”*, no en “probar parámetros al azar”.

---

###  7.4. Selección del mejor modelo

Una vez completada la búsqueda:

- Se evalúa el mejor modelo sobre el conjunto de test.  
- Se compara con los modelos base entrenados previamente.  
- Se documenta si mejora las métricas críticas (especialmente recall y F1-macro).

Esta comparación refuerza la idea de que la optimización es un proceso **técnico, consciente y fundamentado**.

---

###  7.5. Propósito pedagógico del Bloque 7

Este bloque permite al alumnado:

- Entender el impacto real de los hiperparámetros.  
- Practicar un enfoque reproducible a la experimentación.  
- Conectar teoría y práctica con claridad.  
- Adquirir competencias esenciales para roles profesionales de IA/machine learning.  
- Comprender que optimizar un modelo implica también revisar **potenciales sesgos** (no solo maximizar accuracy).

---

###  Conclusión del Punto 7

La comparación de hiperparámetros convierte el Random Forest en un vehículo para aprender:

- razonamiento experimental,  
- análisis riguroso,  
- toma de decisiones técnicas,  
- y pensamiento crítico aplicado a modelos predictivos.

Este bloque añade un gran valor añadido a la PT, demostrando profundidad y madurez técnica.

---

## 8. Mini interfaz con Streamlit desplegada en Hugging Face

Además del notebook, este proyecto incluye una **mini aplicación web interactiva** desarrollada con **Streamlit** y desplegada en **Hugging Face Spaces**.  
Esta interfaz permite explorar los resultados del modelo de forma accesible, sin necesidad de escribir código, y es especialmente útil en contextos formativos con grupos diversos.

Acceso directo a la app:  
https://huggingface.co/spaces/JonasDMR/diabetes-streamlit-jonas

---

###  8.1. ¿Qué permite hacer la mini interfaz?

La app desarrollada en `src/app.py` ofrece:

- Visualizar la **tabla de métricas** de los modelos entrenados (Regresión Logística, Random Forest, etc.).  
- Comparar rápidamente:
  - accuracy,  
  - recall de la clase positiva,  
  - F1-macro.  
- Seleccionar el **modelo** y la **métrica principal** que se quiere explicar.  
- Introducir:
  - el **objetivo del sistema** (ej.: “apoyar la detección temprana del riesgo de diabetes”),  
  - un posible **riesgo de sesgo** (ej.: “menor recall en personas menores de 30 años”).  
- Generar una **explicación en lenguaje accesible**, mediante un **mock de IA generativa (LLM)**, que transforma métricas técnicas en un texto claro y ético.

---

###  8.2. Relación con el notebook (`PT_diabetes.ipynb`)

La mini interfaz está directamente conectada con el trabajo realizado en el notebook:

- El notebook genera el archivo `resultados_modelos.csv` con las métricas clave.  
- La app Streamlit **lee ese CSV** y construye la experiencia visual e interactiva.  

De esta forma, el alumnado puede ver el flujo completo:

1. Preparar datos y entrenar modelos.  
2. Exportar resultados.  
3. Usarlos en una herramienta interactiva.

---

###  8.3. Uso didáctico de la mini interfaz en clase

Algunas formas de usar la app en el aula:

- Como **punto de partida** para alumnado con menos confianza en programación:  
  primero exploran resultados desde la app, luego ven cómo se generó el código.
- Como **herramienta de síntesis** al final de la sesión:  
  para revisar métricas y consolidar conceptos.
- Como **disparador de debate ético**:
  - ¿Qué mensaje damos a una persona si le mostramos esta explicación?  
  - ¿Estamos comunicando bien los límites del modelo?  
  - ¿Qué riesgos hay si alguien confía ciegamente en el sistema?

También puede utilizarse para:

- Evaluaciones prácticas (interpretar lo que muestra la app).  
- Trabajos en grupo (cada grupo elige un modelo y explica sus resultados a la clase).

---

###  8.4. Ventajas de usar Hugging Face Spaces

Desplegar la app en Hugging Face aporta:

- **Accesibilidad:** solo hace falta un navegador, sin instalaciones locales.  
- **Inclusión tecnológica:** ideal para personas sin equipos potentes o sin experiencia en configuración de entornos.  
- **Realismo profesional:** el alumnado ve cómo una solución de IA puede ser expuesta como un servicio web sencillo.  
- **Portafolio:** el propio Space actúa como una demostración pública del trabajo realizado.

---

###  8.5. Propósito pedagógico del Bloque 8

El Bloque 8 refuerza varias competencias clave:

- Entender que un modelo de IA no es solo código, sino parte de una **experiencia de usuario**.  
- Ver cómo se pueden crear herramientas sencillas pero potentes para **acompañar decisiones humanas**, no sustituirlas.  
- Fomentar la **confianza** del alumnado al ver que su trabajo en el notebook puede convertirse en una aplicación real.  
- Mostrar un ejemplo concreto de **IA responsable, explicable y centrada en las personas**.

La mini interfaz es, por tanto, el puente perfecto entre:

- la **parte técnica** (entrenar modelos),  
- la **parte ética** (analizar sesgos y limitaciones),  
- y la **parte pedagógica** (explicar y comunicar IA a otros).

---

## 9. Conclusión

Este proyecto integra:

- un pipeline completo de Machine Learning,
- un análisis explícito de sesgos,
- un mock de IA generativa para explicabilidad accesible,
- y una mini interfaz web para uso en contextos formativos reales.

Está diseñado para ser utilizado en grupos diversos, con distintos niveles de experiencia, y alineado con los valores de inclusión, impacto social y aprendizaje transformador de Fundación Somos F5.






