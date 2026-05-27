# Detección de Cumplimiento de Normas de Seguridad en Obras de Construcción

**Parcial — Deep Learning aplicado a visión por computadora**

---

## Resumen ejecutivo

Se entrenó un detector YOLOv8s para identificar trabajadores de construcción y
los elementos de protección personal (EPP) que portan, y se acopló un módulo
determinista de verificación de cumplimiento. El modelo final alcanza
**mAP@0.5 = 0.771** sobre un conjunto de prueba no visto (3.895 imágenes),
con desempeño excelente en las tres clases críticas (cabeza con casco, cabeza
sin casco y chaleco, todas con mAP@0.5 ≥ 0.947).

Durante el desarrollo se identificó una limitación importante derivada de la
combinación de tres datasets con anotación parcial no superpuesta: el modelo
predice clases distintas según el "estilo" de la imagen, lo que limita la
emisión de veredictos CONFORME cuando se evalúa por trabajador. Esta
limitación se documenta cuantitativamente y se discuten alternativas para
producción.

---

## 1. Introducción

### 1.1 Problema

Las obras de construcción son entornos de alto riesgo. El uso correcto del
casco de seguridad y del chaleco reflectivo es una de las medidas más básicas
para reducir la tasa de accidentes. La supervisión manual del cumplimiento de
estas normas es costosa, intermitente y susceptible a sesgos humanos.

Este trabajo aborda el problema de **detectar automáticamente trabajadores y
clasificar su estado de cumplimiento del EPP** a partir de imágenes y video
de obras, abordando tres desafíos técnicos centrales:

1. **Oclusión severa**: trabajadores parcialmente ocultos por maquinaria,
   andamios u otros trabajadores.
2. **Detección de objetos pequeños**: trabajadores lejanos con pocos píxeles.
3. **Desbalance de clases**: los casos no conformes (la clase crítica) son
   típicamente escasos en grabaciones reales.

### 1.2 Objetivos

- Entrenar y evaluar un modelo YOLO ajustado al dominio de seguridad en obras.
- Implementar un verificador de cumplimiento basado en reglas como
  posprocesamiento determinista.
- Documentar de forma justificada las decisiones de entrenamiento.
- Evaluar cuantitativamente el modelo con desglose por clase y por nivel
  de oclusión.
- Demostrar el sistema sobre video real no visto en entrenamiento.
- Reflexionar críticamente sobre las implicaciones éticas.

### 1.3 Estrategia general

Se adopta una arquitectura híbrida **percepción + decisión**:

- **Percepción**: un modelo YOLOv8 detecta cabezas (con/sin casco), chalecos
  y, cuando es posible, cuerpos completos.
- **Decisión**: un módulo Python independiente toma las detecciones y emite
  un veredicto por trabajador (CONFORME / NO_CONFORME / INDETERMINADO).

Esta separación favorece la **auditabilidad** (la lógica de cumplimiento es
código legible, no parámetros opacos) y la **modificabilidad** (se pueden
cambiar reglas sin reentrenar el modelo).

---

## 2. Datos

### 2.1 Fuentes

Se combinaron tres datasets públicos de Roboflow Universe, descargados en
formato YOLOv8:

| Dataset | Imágenes | Aporta | Licencia |
|---|---:|---|---|
| Hard Hat Workers (Northeastern) | 7.035 | cabezas con/sin casco + person | Public Domain |
| Safety Helmet (lisa-group-de) | 7.035 | cabezas con/sin casco + person | CC BY 4.0 |
| Safety Vest PPE (big-dataset-ppe) | 11.906 | chalecos reflectivos | CC BY 4.0 |
| **Total** | **25.976** | | |

### 2.2 Taxonomía unificada

Las clases de cada dataset original se remapearon a una taxonomía común de
5 clases mediante una tabla declarativa versionada (`configs/class_mapping.yaml`):

| id | Clase | Significado |
|---|---|---|
| 0 | head_with_helmet | Cabeza con casco visible |
| 1 | head_without_helmet | Cabeza sin casco (clase crítica) |
| 2 | person | Cuerpo completo o parcial del trabajador |
| 3 | vest | Chaleco reflectivo visible |
| 4 | no_vest_person | Persona sin chaleco identificable |

### 2.3 Splits estratificados

Se generaron splits **70/15/15** train/val/test estratificados por
**combinación de clases presentes en cada imagen** (no por bbox). Esto
preserva la proporción de cada clase en cada split y previene la fuga de
datos entre splits, una condición esencial para que la métrica final sobre
test sea honesta.

Distribución de instancias por split:

| Clase | train | val | test | total | % del total |
|---|---:|---:|---:|---:|---:|
| head_with_helmet | 32.055 | 6.974 | 7.222 | 46.251 | 56.3% |
| head_without_helmet | 10.645 | 2.194 | 2.104 | 14.943 | 18.2% |
| vest | 13.217 | 2.828 | 2.831 | 18.876 | 23.0% |
| person | 1.116 | 282 | 216 | 1.614 | 2.0% |
| no_vest_person | 0 | 0 | 0 | 0 | 0% |

### 2.4 Reconocimiento de límites del dataset

Dos limitaciones se identifican explícitamente:

1. **`person` muy escasa** (2.0% de las instancias): los tres datasets son
   detectores centrados en EPP, no anotan exhaustivamente el cuerpo completo
   del trabajador. La consecuencia, observada en evaluación, es que la clase
   `person` arrastra el mAP global hacia abajo y su detección en escena no
   es confiable.

2. **`no_vest_person` ausente**: ningún dataset aporta esa etiqueta. La
   ausencia de chaleco se **infiere por reglas espaciales** en el verificador
   (cabeza detectada sin chaleco asociado debajo).

3. **Anotación parcial no superpuesta**: Hard Hat y SHWD anotan solo cabezas;
   el dataset de PPE anota solo chalecos. El modelo aprendió a predecir clases
   distintas según el "estilo" de la imagen. Esta limitación se discute en
   profundidad en la sección 5.

---

## 3. Metodología

### 3.1 Elección del modelo

Se eligió **YOLOv8** (Ultralytics) por las siguientes razones:

- **Madurez del tooling**: API estable, documentación extensa, soporte
  comunitario activo.
- **Compatibilidad probada** con Google Colab y Kaggle Notebooks.
- **Variantes de tamaño**: se usó YOLOv8s (small, ~11M parámetros) para
  todos los experimentos, balanceando capacidad y tiempo de entrenamiento.
- **Augmentaciones nativas**: cubre las necesidades del problema (oclusión,
  variabilidad de escala, robustez a iluminación) sin librerías adicionales.

### 3.2 Configuración base (baseline)

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Resolución de entrada | 640×640 | Default de Ultralytics, balance ajuste/velocidad |
| Epochs | 80 | Permite convergencia sin early stopping prematuro |
| Patience | 20 | Early stopping si no mejora en 20 epochs |
| Optimizador | SGD | Default recomendado de Ultralytics |
| Learning rate inicial | 0.01 | Estándar |
| Batch | 16 | Maximiza uso de VRAM en T4 |
| Mosaic | 1.0 | Default, importante para diversidad |
| MixUp | 0.0 | Desactivado en baseline |
| Random Erasing | 0.0 | Desactivado en baseline |

### 3.3 Experimentos comparativos

Se diseñaron cuatro experimentos para atacar los tres desafíos del enunciado.
Por restricciones de cuota GPU (Kaggle limita a 30 h/semana, cada
entrenamiento toma ~7-8 horas) se ejecutaron 2 de 4, con justificación para
los descartes:

| Exp | Cambio respecto al baseline | Objetivo |
|---|---|---|
| A | Resolución 960 | Mejorar detección de objetos pequeños |
| **B** | **mixup=0.15, erasing=0.4** | **Robustez a oclusión** |
| C | Pesos de clase / oversampling | Atacar desbalance de clases |
| **D** | **close_mosaic=15** | **Afinar el modelo con imágenes en distribución natural en epochs finales** |

**A se descartó** por riesgo de timeout en Kaggle (∼12-15 h estimadas, sobre
el límite de 12 h por sesión) y costo de oportunidad alto vs. mejora marginal
esperada en un dataset ya limpio.

**C se descartó** porque el problema de la clase `person` es de **cantidad
de datos** (1.116 train), no de balance: ningún esquema de pesos
"inventa" más ejemplos.

### 3.4 Decisiones técnicas justificadas

**Anchors**: Ultralytics 8.x usa estrategia *anchor-free* (DFL — Distribution
Focal Loss). No hay anchors a configurar; el modelo predice la distribución
del bbox directamente.

**Augmentaciones**: estrictamente las nativas de Ultralytics (HSV jitter,
flip horizontal, translación, escala, mosaic). El Experimento B añadió
Random Erasing y MixUp moderado.

**Mosaic**: activo durante 65 de 80 epochs en el modelo final (Experimento D),
desactivado en los últimos 15 epochs (`close_mosaic=15`) para afinar con
imágenes en su distribución natural.

**Resolución**: 640 en todos los experimentos. Subir a 960 era el plan original
pero se descartó por tiempo.

### 3.5 Stack tecnológico

- Python 3.11, PyTorch 2.x, Ultralytics 8.4.x.
- OpenCV (lectura de video y rendering), Supervision (anotación de bboxes).
- NumPy, Pandas, PyYAML, Matplotlib (procesamiento y reportes).
- pytest (39 tests unitarios en el módulo de cumplimiento).
- Git para versionado de código y configs; Kaggle Datasets para los datos
  remapeados; Kaggle Notebooks para entrenamiento e inferencia pesada.

---

## 4. Resultados cuantitativos

### 4.1 Comparación de experimentos (sobre validación)

| Clase | Baseline | Exp B | **Exp D** |
|---|---:|---:|---:|
| head_with_helmet (mAP@0.5) | 0.978 | 0.978 | **0.979** |
| head_without_helmet (mAP@0.5) | 0.959 | 0.960 | 0.959 |
| vest (mAP@0.5) | 0.987 | 0.986 | 0.986 |
| person (mAP@0.5) | 0.175 | 0.153 | 0.175 |
| **all mAP@0.5** | **0.775** | 0.769 | **0.775** |
| **all mAP@0.5-0.95** | 0.567 | 0.562 | **0.569** |

**Modelo elegido: Experimento D** (`close_mosaic=15`). Mejora consistente
en mAP@0.5-0.95 en todas las clases (+0.2 pp).

**Hallazgo del Experimento B**: las augmentaciones agresivas (Random Erasing
+ MixUp) **no aportaron** mejora en este dataset; al contrario, degradaron
ligeramente la precisión global (−0.6 pp). Esto sugiere que los datos ya
contenían suficiente variabilidad natural y el ruido sintético adicional
no era beneficioso. Hallazgo válido para reportar honestamente.

### 4.2 Evaluación final sobre el conjunto de prueba

Modelo D evaluado sobre **3.895 imágenes nunca vistas** durante el
entrenamiento ni la selección de hiperparámetros.

| Clase | Instancias | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|---|---:|---:|---:|---:|---:|
| head_with_helmet | 7.222 | 0.939 | 0.950 | **0.976** | 0.664 |
| head_without_helmet | 2.104 | 0.892 | 0.942 | **0.947** | 0.646 |
| vest | 2.831 | 0.937 | 0.957 | **0.984** | 0.837 |
| person | 216 | 0.341 | 0.208 | **0.176** | 0.129 |
| **all** | 12.373 | **0.777** | **0.765** | **0.771** | **0.569** |

### 4.3 Generalización (val vs test)

| Métrica | Val | Test | Δ | Interpretación |
|---|---:|---:|---:|---|
| mAP@0.5 | 0.775 | 0.771 | −0.004 | Sin overfitting |
| mAP@0.5-0.95 | 0.569 | 0.569 | 0.000 | Idéntico |
| Precision | 0.795 | 0.777 | −0.018 | Caída mínima |
| Recall | 0.765 | 0.765 | 0.000 | Idéntico |

El modelo generaliza bien. La estratificación sin fuga del Fase 1 funcionó.

### 4.4 Análisis de robustez por nivel de oclusión

Para cada imagen del test se calculó el **IoU máximo entre cualquier par
de cajas anotadas** y se clasificó en tres niveles. Procedimiento detallado
en `src/analysis/occlusion.py`.

#### Distribución

| Nivel | IoU máximo | Imágenes | % del test |
|---|---|---:|---:|
| Bajo (low) | < 0.15 | 3.695 | 94.9% |
| Medio (medium) | [0.15, 0.40) | 185 | 4.7% |
| Alto (high) | ≥ 0.40 | 15 | 0.4% |

#### Métricas por nivel

| Nivel | n | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Bajo | 3.695 | **0.779** | 0.576 | 0.782 | **0.796** |
| Medio | 185 | 0.789 | 0.565 | 0.797 | 0.749 |
| Alto | 15 | **0.704** | 0.512 | 0.783 | **0.623** |
| **Caída bajo → alto** | — | **−7.5 pp** | −6.4 pp | +0.1 pp | **−17.3 pp** |

#### Lectura

- **El modelo es robusto a oclusión leve-media**: low y medium tienen
  mAP@0.5 prácticamente equivalente.
- **Oclusión alta sí degrada el desempeño**: caída de 7.5 pp en mAP@0.5.
- **El recall es la métrica más afectada** (−17.3 pp), no la precisión.
  Esto significa que el modelo **pierde detecciones** (no las ve) cuando hay
  oclusión fuerte, pero las que sí encuentra siguen bien clasificadas.
  Operativamente: en escenas concurridas se podrían perder trabajadores
  no conformes (falso negativo crítico para seguridad).
- **Caveat estadístico**: el grupo "alto" tiene n=15. El estimador tiene
  varianza considerable; la tendencia es clara pero el valor exacto del
  "−7.5 pp" tiene un intervalo de confianza amplio.

---

## 5. Verificador determinista de cumplimiento

### 5.1 Diseño

El verificador es un **módulo Python independiente** que consume las
detecciones de YOLO por fotograma y emite un veredicto por trabajador.
Está implementado en `src/compliance/` con tres archivos:

- `geometry.py`: utilidades puras (IoU, contención, asociación espacial).
- `verifier.py`: lógica de decisión.
- `renderer.py`: dibujo de bboxes coloreados.

Cuenta con **15 tests unitarios** sobre escenarios sintéticos y se valida
con 14 tests adicionales del módulo de geometría.

### 5.2 Estrategia de asociación: cabeza-ancla

El diseño inicial usaba `person` como ancla ("para cada persona detectada,
buscar cabeza y chaleco asociados"). En evaluación cualitativa este diseño
falló porque la clase `person` es muy escasa (recall del 17% sobre val).

Se reescribió la lógica como **cabeza-ancla**: para cada cabeza detectada,
se busca un chaleco debajo con alineación horizontal razonable. Si existe
un `person` que contiene la cabeza, se usa solo para enriquecer el rendering
(bbox grande del trabajador). Esta versión opera correctamente con las
detecciones reales del modelo.

### 5.3 Lógica de veredicto

| Detecciones asociadas a un trabajador | Veredicto |
|---|---|
| cabeza con casco + chaleco debajo | **CONFORME** |
| cabeza con casco, sin chaleco | NO_CONFORME ("chaleco_no_visible") |
| cabeza sin casco | NO_CONFORME ("casco_ausente") |
| chaleco huérfano (sin cabeza visible) | INDETERMINADO ("cabeza_no_visible") |

La separación de INDETERMINADO de NO_CONFORME es deliberada: solo se
penaliza cuando hay evidencia explícita de falta, no cuando simplemente
falta información para juzgar.

### 5.4 Validación cualitativa sobre val real

Se ejecutó el pipeline completo (YOLO + verificador) sobre 200 imágenes
del val set y se inspeccionaron los resultados manualmente. Hallazgos:

- El verificador **funciona correctamente** cuando recibe detecciones de
  ambas clases (cabeza + chaleco).
- Sobre 200 imágenes: 9 CONFORME, 599 NO_CONFORME, 149 INDETERMINADO.
- **El cuello de botella NO es el verificador, es el modelo**.

### 5.5 Hallazgo principal: sesgo de etiquetado entre datasets

Sobre 300 imágenes val analizadas:

| Categoría | % |
|---|---:|
| Solo `head_with_helmet` detectado | 49.0% |
| Solo `head_without_helmet` detectado | 7.3% |
| Solo `vest` detectado | 40.3% |
| `head` + `vest` (compatible con veredicto CONFORME) | **3.3%** |

El modelo casi nunca firma cabeza Y chaleco en la misma imagen porque
los datasets de entrenamiento están etiquetados parcialmente:
SHWD/Hard Hat anotan solo cabezas, PPE anota solo chalecos. El modelo
aprendió la correlación "estilo de imagen → clases visibles" y la
internalizó como un comportamiento bimodal.

**Esta es una limitación del dataset, no de la lógica del verificador.**
Para producción, se recomienda re-entrenar sobre un dataset con anotación
completa por trabajador (clases Hardhat + Safety Vest + Person presentes
en cada imagen, ej. Construction Site Safety).

---

## 6. Demostración en video

### 6.1 Material

Se procesó un video de obra real **no visto durante el entrenamiento**:
una cuadrilla colombiana vertiendo concreto desde un camión Argos en una
calle urbana. 25.8 segundos a 30 fps (773 frames) en 1920×1080.

### 6.2 Configuración de inferencia

- Modelo: `runs_outputs/exp_D/best.pt` (YOLOv8s, Experimento D).
- Resolución de inferencia: 1280 (subida desde el 640 de entrenamiento para
  mejorar la detección de cabezas, que son objetos más chicos a esta
  distancia de cámara).
- Umbral de confianza: 0.15 (más permisivo que el 0.25 default; mejora la
  recall a costa de algunos falsos positivos aceptables para una demo).

### 6.3 Resultados agregados

| Veredicto | Cantidad sobre 773 frames |
|---|---:|
| CONFORME | 861 |
| NO_CONFORME | 2.600 |
| INDETERMINADO | 1.309 |
| **Tasa de cumplimiento (excluyendo INDETERMINADO)** | **24.9%** |

### 6.4 Observaciones cualitativas

- Los tres tipos de veredicto aparecen claramente en distintos fotogramas.
- La tasa de cumplimiento se actualiza por fotograma (visible en la
  esquina superior izquierda del video).
- El verificador reporta razones específicas en los rótulos:
  "chaleco_no_visible", "casco_ausente", "cabeza_no_visible".
- La limitación documentada en sección 5.5 se confirma en datos reales:
  muchos chalecos quedan como INDETERMINADO porque el modelo no detecta
  las cabezas asociadas.

### 6.5 Archivo

`demo_video/video03_annotated_v2.mp4` (114 MB, MP4 H.264).

### 6.6 Validación experimental complementaria (modelo externo)

Para verificar si la alta proporción de INDETERMINADO observada se debía
al **pipeline** (verificador, asociación) o al **dataset de entrenamiento**
(anotación parcial), se ejecutó el mismo pipeline sustituyendo nuestro
modelo D por un modelo YOLOv8 externo entrenado sobre un dataset con
anotación completa por trabajador (10 clases: Hardhat, NO-Hardhat,
Safety Vest, NO-Safety Vest, Person, Mask, etc.). El verificador y el
renderer **no se modificaron**; solo se remapean las clases del modelo
externo a la taxonomía unificada antes de pasar al verificador.

Resultados sobre el mismo video (`video03p.mp4`, 773 frames):

| Métrica | Modelo D (nuestro) | Modelo externo | Ratio |
|---|---:|---:|---:|
| CONFORME | 861 | 2.287 | **2.66×** |
| NO_CONFORME | 2.600 | 7.221 | **2.78×** |
| INDETERMINADO | 1.309 | 397 | **0.30×** ↓ |
| Veredictos determinados | 3.461 | 9.508 | **2.75×** |
| INDETERMINADO sobre el total | 27.4% | **4.0%** | −85% |
| Tasa de cumplimiento (CONF/determinados) | 24.9% | 24.0% | similar |

**Lectura**: el verificador funciona idénticamente en ambos casos — no se
tocó una línea de su código. La diferencia abrumadora se debe a **qué
detecciones recibe**. Cuando el detector tiene anotaciones completas en
su entrenamiento, las clases coexisten en cada frame y el sistema produce
2.75× más veredictos definitivos sobre la misma escena.

**Esto confirma cuantitativamente la hipótesis** de la sección 5.5: la
limitación operativa observada con nuestro modelo D es **estructural del
dataset** combinado (anotación parcial no superpuesta), **no del pipeline**.

> **Aclaración de honestidad académica**: el modelo externo no es producto
> de este trabajo. Se usó únicamente como referencia experimental para
> validar nuestro diagnóstico. El modelo principal entregado sigue siendo
> el Experimento D entrenado en este parcial, con sus métricas reportadas
> honestamente.

---

## 7. Implicaciones éticas

### 7.1 Balance entre falsos negativos y falsos positivos

Los dos tipos de error tienen costos asimétricos:

- **Falso negativo** (no detectar un trabajador sin EPP): el caso queda sin
  reportar, el trabajador permanece en riesgo. Si ese trabajador sufre un
  accidente, hay responsabilidad legal y, más importante, daño humano.
- **Falso positivo** (marcar un trabajador conforme como no conforme): el
  trabajador es sancionado o requerido a corregir algo que no estaba mal.
  Esto erosiona la confianza en el sistema, puede generar resistencia o
  sabotaje, y consume tiempo de supervisión.

**En seguridad ocupacional, el falso negativo es típicamente más costoso**
que el falso positivo, porque el riesgo de daño físico es asimétrico. Sin
embargo, una tasa alta de falsos positivos también compromete el sistema:
si los trabajadores aprenden que el sistema "se equivoca seguido", dejarán
de confiar en él incluso cuando acierte.

**Implicación para el umbral operativo**: el umbral de confianza del
clasificador (`conf` en la inferencia) **no debe elegirse maximizando el
mAP global**, sino considerando el costo asimétrico de los errores. Una
recomendación práctica es elegir el umbral que maximiza recall sobre las
clases críticas (`head_without_helmet`, `no_vest_person`) sujeto a un techo
de falsos positivos tolerables. La curva precision-recall por clase debería
guiar esta elección.

**Implicación para el verificador**: el estado INDETERMINADO es un
mecanismo de seguridad importante. El sistema no debe **inventar veredictos**
cuando no tiene evidencia suficiente; debe escalar la decisión a un humano.
Esto es lo que hace el verificador propuesto.

### 7.2 Privacidad de los trabajadores

Un sistema que graba continuamente a los trabajadores plantea cuestiones
serias de privacidad:

- **Vigilancia continua**: el monitoreo permanente, aunque sea por motivos
  de seguridad, puede generar estrés, sensación de desconfianza por parte
  del empleador y desincentivar el reporte voluntario de incidentes.
- **Uso secundario de las imágenes**: una vez grabadas, las imágenes pueden
  ser reutilizadas para fines no acordados (evaluación de desempeño, control
  de ritmos de trabajo, identificación de personas en otros contextos). La
  política de retención y borrado de datos debe ser explícita.
- **Consentimiento informado**: los trabajadores deben saber que están
  siendo monitoreados, con qué propósito, dónde se almacenan los datos y
  por cuánto tiempo. El consentimiento bajo coerción laboral (firmá o
  no trabajás) no es consentimiento válido.

**Recomendaciones operativas**: anonimización de rostros a nivel de pipeline,
retención mínima necesaria (ej. solo los frames donde se detecta un evento,
no la grabación completa), acceso restringido a roles específicos, auditoría
de uso. El sistema técnico que proponemos no implementa estas medidas; son
necesarias para un despliegue ético.

### 7.3 Sesgos introducidos por la distribución de los datos

Los tres datasets utilizados tienen sesgos identificables:

- **Sesgo geográfico**: los datasets de Roboflow Universe suelen estar
  dominados por imágenes de obras en China, EE.UU. y Europa. Trabajadores
  con vestimenta tradicional, condiciones climáticas extremas o herramientas
  locales pueden estar subrepresentados. Un modelo desplegado en una región
  con distribución distinta puede tener desempeño degradado sin que se note
  hasta que falle.
- **Sesgo demográfico**: la distribución de género, edad y etnicidad de
  los trabajadores en los datasets no es conocida ni reportada. Un modelo
  puede tener desempeño peor en grupos subrepresentados (ej. trabajadoras
  mujeres, en algunos contextos minoritarias en obras). No se ha auditado
  este eje en este trabajo y se reconoce explícitamente como una limitación.
- **Sesgo de condiciones de captura**: iluminación diurna (la mayoría de
  los datos), clima despejado, ángulos de cámara fijos. El modelo puede
  fallar con luz tenue, lluvia, nieve, o cámaras móviles con desenfoque.

**Implicación**: las métricas reportadas (mAP@0.5 = 0.771 en test) son
**válidas para la distribución del test**, que es muy similar a la
distribución de entrenamiento. **No son garantía** de desempeño equivalente
en condiciones de despliegue distintas. Para despliegue real debería
hacerse un piloto controlado en el sitio objetivo, con re-evaluación.

---

## 8. Limitaciones y trabajo futuro

### 8.1 Limitaciones reconocidas

1. **Sesgo de etiquetado entre datasets**: el modelo tiende a predecir
   clases distintas según el "estilo" de la imagen. Solo el 3.3% de las
   imágenes val tiene cabeza y chaleco detectados simultáneamente.
2. **Clase `person` escasa**: 1.116 instancias en train, mAP@0.5 ≈ 0.18
   sobre test. La detección de cuerpo completo no es confiable.
3. **No se evaluó el Experimento A (resolución 960)** por restricción de
   tiempo de cómputo en Kaggle (límite de 12h por sesión).
4. **Grupo "oclusión alta" estadísticamente débil** (n=15 imágenes).
5. **Sin auditoría de sesgos demográficos** sobre los datasets.
6. **Sin tracking temporal**: cada frame se procesa de forma independiente;
   los veredictos pueden parpadear entre frames si la detección oscila.

### 8.2 Trabajo futuro

- **Dataset con anotación completa**: re-entrenar sobre un único dataset
  bien anotado por trabajador (ej. Construction Site Safety con clases
  Hardhat + Safety Vest + Person presentes en cada imagen).
- **Modelo más grande**: YOLOv8m o YOLOv9 si la cuota de cómputo lo
  permite.
- **Resolución 960 o 1280**: para mejorar detección de objetos lejanos.
- **Tracking temporal** (ByteTrack o Supervision Tracker) para estabilizar
  el veredicto entre frames consecutivos.
- **Calibración de umbral operativo** según el costo asimétrico de errores.
- **Detección de arnés** para trabajos en altura (no implementada en este
  alcance, mencionada como contextual en el enunciado).

---

## 9. Conclusiones

1. Se entrenó un detector YOLOv8s con **mAP@0.5 = 0.771** sobre test no
   visto, con desempeño excelente en las tres clases críticas
   (cabezas con/sin casco y chalecos, todas ≥ 0.947).

2. El **modelo generaliza correctamente**: las métricas test son
   prácticamente idénticas a val (−0.4 pp), validando la calidad de
   los splits estratificados.

3. El **verificador determinista funciona** cuando recibe las detecciones
   esperadas: se validó con 29 tests unitarios y con inspección
   cualitativa sobre val y video real.

4. La principal **limitación operativa es del modelo, no del verificador**:
   el sesgo de etiquetado entre los tres datasets hace que solo el 3.3%
   de las imágenes tengan cabeza Y chaleco detectados simultáneamente.
   Esta limitación se documentó cuantitativamente y **se validó
   experimentalmente** comparando contra un modelo externo entrenado sobre
   un dataset con anotación completa: ese modelo produce 2.75× más
   veredictos definitivos sobre la misma escena (sección 6.6), confirmando
   que el problema es del dataset y no del pipeline.

5. El sistema **expone honestamente su incertidumbre** mediante el estado
   INDETERMINADO, lo que es operativamente correcto: el sistema no debe
   inventar veredictos cuando le falta evidencia.

6. La sección ética identifica tres ejes críticos para despliegue real:
   asimetría de errores (FN vs FP), privacidad y sesgos del dataset.
   El sistema técnico desarrollado **no resuelve** estos ejes por sí
   solo; requieren políticas operativas y de gobernanza.

---

## Apéndice A: Reproducibilidad

### A.1 Estructura del repositorio

```
epp-yolo-compliance/
├── configs/                # data.yaml, hyp_*.yaml, class_mapping.yaml
├── data/                   # gitignored (datasets se descargan manualmente)
├── notebooks/kaggle/       # 01-08 notebooks orquestadores
├── reports/                # tablas y figuras del informe
├── src/
│   ├── analysis/           # análisis de oclusión
│   ├── compliance/         # geometría + verificador + renderer
│   └── data/               # remap, splits, validación
├── tests/                  # 39 tests pytest
├── pyproject.toml
├── requirements.txt
└── README.md
```

### A.2 Comando para reproducir Fase 1 (preparación de datos)

```bash
# 1. Remapear cada dataset a la taxonomía unificada
python -m src.data.remap_labels --dataset helmet --src data/raw/helmet --dst data/interim/helmet
python -m src.data.remap_labels --dataset ppe    --src data/raw/ppe    --dst data/interim/ppe
python -m src.data.remap_labels --dataset shwd   --src data/raw/shwd   --dst data/interim/shwd

# 2. Validar visualmente
python -m src.data.validate_visually --src data/interim/helmet --n 100 --out reports/figures/val_helmet
# (repetir para ppe y shwd)

# 3. Splits estratificados 70/15/15
python -m src.data.make_splits --src data/interim --dst data/processed --ratios 0.70 0.15 0.15

# 4. Reporte de distribución
python -m src.data.class_distribution_report --src data/processed --out reports/tables/class_distribution.csv
```

### A.3 Notebooks de entrenamiento y evaluación (Kaggle)

| Notebook | Propósito | Tiempo aproximado |
|---|---|---:|
| `01_baseline_train.ipynb` | Baseline YOLOv8s @ 640 | ~7.3 h |
| `03_exp_B_augmentation.ipynb` | Random Erasing + MixUp | ~7.9 h |
| `05_exp_D_close_mosaic.ipynb` | close_mosaic=15 (modelo final) | ~7.2 h |
| `07_test_evaluation.ipynb` | Eval sobre test set | ~4 min |
| `08_occlusion_analysis.ipynb` | mAP por nivel de oclusión | ~5 min |

### A.4 Pipeline de inferencia local

```bash
# Inferencia + verificador sobre imágenes
python -m src.run_pipeline --weights runs_outputs/exp_D/best.pt \
    --images data/processed/images/test --out reports/figures/demo --n 20

# Inferencia + verificador sobre video
python -m src.run_video --weights runs_outputs/exp_D/best.pt \
    --video demo_video/video03p.mp4 --out demo_video/video03_annotated_v2.mp4 \
    --conf 0.15 --imgsz 1280
```

### A.5 Artefactos publicados

- **Código**: `https://github.com/PercyTechX/epp-yolo-compliance`.
- **Dataset remapeado** (Kaggle): `jpnazariop/epp-yolo-processed`.
- **Pesos del modelo final** (Kaggle): `jpnazariop/epp-yolo-exp-d-weights`.
- **Video demo**: `demo_video/video03_annotated_v2.mp4` (publicación pendiente).
