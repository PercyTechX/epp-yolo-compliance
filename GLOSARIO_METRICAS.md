# Glosario de métricas para la exposición

Documento de estudio para la defensa del parcial. Cada métrica viene
con: (1) qué mide, (2) cómo se calcula en términos simples, (3) el
número real de nuestro proyecto y (4) cómo defenderla si el profesor
pregunta.

---

## 1. Métricas de detección de objetos

### 1.1 IoU — Intersection over Union

**Qué mide:** qué tanto se solapa una caja predicha por el modelo con la
caja real (la "verdad", que está en las anotaciones).

**Cómo se calcula:**
```
IoU = Área de intersección / Área de unión
```
- Si las cajas son idénticas → IoU = 1.0 (100%)
- Si no se tocan → IoU = 0
- Típico para considerar "buena detección": IoU ≥ 0.5

**Para qué sirve:** decidir si una predicción "acertó" o no. Un IoU bajo
significa que el modelo detectó algo pero el bbox está mal ubicado.

---

### 1.2 Precision (precisión)

**Qué mide:** de todo lo que el modelo predijo como "clase X", **¿cuánto
era realmente clase X?**

**Fórmula:**
```
Precision = Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)
```

**Analogía:** sos detector de fraudes en un banco. Marcaste 100 transacciones
como sospechosas. Si 80 eran fraudes reales → precision = 80%.

**Nuestro número (test, modelo D):**
- head_with_helmet: **0.939** (93.9% de las veces que dijo "casco", era casco)
- vest: **0.937**
- person: **0.341** (la mayoría de "person" predichos no eran "person" reales)

**Cómo defenderlo:** "Precision alta = pocos falsos positivos = no alarmamos
sin razón."

---

### 1.3 Recall (sensibilidad)

**Qué mide:** de todos los casos REALES de clase X, **¿cuántos detectó
el modelo?**

**Fórmula:**
```
Recall = Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)
```

**Analogía:** mismo banco, 100 transacciones eran fraudes. Marcaste 70.
Recall = 70%. (Te perdiste 30.)

**Nuestro número (test, modelo D):**
- head_with_helmet: **0.950** (atrapa el 95% de cabezas con casco que hay)
- vest: **0.957**
- person: **0.208** (atrapa solo el 21% de personas reales — clase escasa)

**Cómo defenderlo:** "Recall alto = pocos falsos negativos = no se nos
escapan casos. **En seguridad ocupacional, recall importa más que precision**
porque un falso negativo significa no detectar a un trabajador en riesgo."

---

### 1.4 Precision vs Recall — el trade-off

| Estrategia | Resultado |
|---|---|
| Modelo muy "tímido" (solo predice cuando está super seguro) | Precision alta, recall bajo |
| Modelo muy "agresivo" (predice ante la mínima sospecha) | Recall alto, precision baja |

**Se ajusta con el umbral de confianza (`conf`)**: bajo → más recall,
alto → más precision.

---

### 1.5 mAP — mean Average Precision

**Qué mide:** una métrica resumen que combina precision y recall para
cada clase, promediada a lo largo de muchos umbrales.

**Cómo se calcula** (simplificado):
1. Para cada clase, dibujás la curva precision vs recall.
2. Calculás el área bajo esa curva → eso es AP (Average Precision).
3. Promediás los AP de todas las clases → eso es **mAP**.

**Rango:** 0 (peor) a 1 (perfecto).

**Equivalente intuitivo:** "Si tomo todas las predicciones del modelo y las
ordeno por confianza, ¿qué proporción son buenas a distintos niveles de
agresividad?"

---

### 1.6 mAP@0.5

**Qué mide:** mAP considerando como "acierto" todo bbox con IoU ≥ 0.5
respecto del bbox real.

**Nuestro número:** **0.771** sobre el test set (todas las clases).

**Cómo defenderlo:** "Es la métrica estándar de la industria para
comparar detectores. 0.771 significa que el modelo es razonablemente
bueno detectando objetos con bbox bien ubicados."

---

### 1.7 mAP@0.5-0.95 (COCO mAP)

**Qué mide:** lo mismo que mAP@0.5 pero promediado sobre 10 umbrales de
IoU (0.50, 0.55, 0.60, …, 0.95). Más estricto.

**Por qué es más exigente:** a IoU=0.95, el bbox tiene que estar
casi perfectamente alineado con el real. La mayoría de los detectores
fallan ahí.

**Nuestro número:** **0.569** sobre el test set.

**Cómo defenderlo:** "Es la métrica más estricta de evaluación porque
exige no solo detectar, sino ubicar el bbox con precisión. 0.569 es un
nivel sólido para YOLOv8s en su tamaño base."

---

### 1.8 F1 Score

**Qué mide:** media armónica entre precision y recall.

**Fórmula:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Para qué sirve:** balancear precision y recall en un solo número. Útil
cuando ambos importan por igual.

**Nuestro proyecto:** no se usa directamente como métrica principal pero
sí aparece en las curvas `BoxF1_curve.png` que generó Ultralytics.

---

### 1.9 Confusion Matrix (matriz de confusión)

**Qué muestra:** una grilla de "lo que predijo el modelo" vs "lo que era
en realidad".

- **Diagonal**: aciertos (clase X predicha como clase X). Fuerte = bueno.
- **Fuera de diagonal**: confusiones entre clases. Idealmente cero o
  bajo.
- **Columna "background"**: falsos negativos (se le escaparon).
- **Fila "background"**: falsos positivos (vio cosas que no existían).

**Nuestro resultado** (Fig. 3 del informe): la diagonal es muy fuerte
y casi no hay confusión entre clases. La mayoría de errores caen en
"background" — son detecciones perdidas, no confusiones de clase.

**Cómo defenderlo:** "La matriz muestra que el modelo distingue las 4
clases sin confundirse entre ellas. Los errores son por detección
perdida (falso negativo), no por mala clasificación."

---

## 2. Métricas de entrenamiento (las que ves en results.png)

### 2.1 Box loss (pérdida de localización)

**Qué mide:** qué tan mal el modelo ubica los bboxes.

**Tendencia esperada:** debe **bajar** epoch tras epoch. Si sube o
se queda plana, algo va mal.

**Nuestro proyecto:** baja monotónicamente de 1.35 (epoch 1) a 0.86
(epoch 80). Tendencia saludable.

---

### 2.2 Cls loss (pérdida de clasificación)

**Qué mide:** qué tan mal el modelo etiqueta lo que detecta (asignar
"casco" vs "cabeza sin casco").

**Tendencia esperada:** **baja**.

**Nuestro proyecto:** de 1.27 a 0.42 en 80 epochs.

---

### 2.3 DFL loss (Distribution Focal Loss)

**Qué mide:** qué tan precisa es la "distribución" que predice el modelo
para cada coordenada del bbox. Es propia de YOLOv8 (no aparece en
YOLOv5 o anteriores).

**Tendencia esperada:** **baja**.

**Cómo defenderlo:** "Es una pérdida específica de YOLOv8 que reemplazó
los anchors clásicos. El modelo predice una distribución de probabilidad
sobre las coordenadas en vez de un valor único."

---

### 2.4 Curvas saludables vs problemáticas

| Patrón en los gráficos | Diagnóstico |
|---|---|
| Train loss baja, val loss baja | ✅ Aprendiendo bien |
| Train baja, val sube | ❌ Overfitting (memorizando train) |
| Ambas planas | ❌ Underfitting / atascado |
| mAP sube monótono | ✅ Mejora real |
| mAP oscila violentamente | ⚠️ Inestabilidad, posible learning rate alto |

**Nuestro caso:** train y val bajan ambos sin separarse, mAP sube
monótono. **Sin overfitting, sin instabilidad.**

---

## 3. Veredictos del verificador (parte propia del proyecto)

### 3.1 CONFORME 🟢

**Cuándo se asigna:** cuando se detecta una cabeza CON casco Y un
chaleco asociado debajo de ella.

**Significado:** el trabajador cumple con el EPP requerido.

---

### 3.2 NO_CONFORME 🔴

**Cuándo se asigna:**
- `casco_ausente`: se detectó una cabeza pero sin casco.
- `chaleco_no_visible`: hay cabeza con casco pero no se detectó chaleco.
- `casco_ausente + chaleco_no_visible`: ambos faltan.

**Significado:** el trabajador NO cumple con el EPP requerido (al menos
desde lo que el sistema puede observar).

---

### 3.3 INDETERMINADO 🟡

**Cuándo se asigna:** se detectó un chaleco huérfano (sin cabeza
asociada arriba). El sistema no puede juzgar sin saber si esa persona
lleva casco.

**Significado:** "no tengo evidencia suficiente para decidir". Es un
mecanismo de seguridad: **el sistema prefiere admitir incertidumbre
antes que inventar un veredicto**.

**Cómo defenderlo:** "El estado INDETERMINADO es una decisión de diseño,
no un error. Cuando el modelo no provee evidencia suficiente, el
verificador escala la decisión a un humano en vez de inventar un fallo
o aprobación. Esto es coherente con el principio ético de que un
sistema de seguridad no debe alucinar resultados."

---

### 3.4 Tasa de cumplimiento

**Fórmula:**
```
Tasa = CONFORME / (CONFORME + NO_CONFORME)
```

Los INDETERMINADO **NO cuentan** en el denominador. Solo se calcula
sobre los veredictos definitivos.

**Por qué excluir INDETERMINADO:** si tuvieras 100 trabajadores y 50
INDETERMINADO, no querés que esos 50 cuenten como "no conformes" en la
tasa porque no sabés.

---

## 4. Conceptos sobre datos

### 4.1 Train / Val / Test split

**Train (entrenamiento):** las imágenes que el modelo VE y de las que
aprende. ~70% del dataset.

**Val (validación):** imágenes que el modelo NO ve durante el
entrenamiento pero que se usan para medir cada cierto tiempo si está
mejorando. Sirve para elegir hiperparámetros y modelos. ~15%.

**Test (prueba):** imágenes que el modelo **NUNCA** ve hasta el final.
Es la prueba honesta de qué tan bien generaliza. ~15%.

**Por qué importa la separación:** si usás test durante el desarrollo,
"infiltrás" información y las métricas se vuelven optimistas. El test
debe ser sagrado hasta el final.

**Nuestros números:** train = 18.184 imgs, val = 3.897, test = 3.895.

---

### 4.2 Stratified split (split estratificado)

**Qué es:** repartir las imágenes a train/val/test cuidando que la
proporción de cada clase sea **la misma en cada split**.

**Por qué importa:** si por azar todas las imágenes de "head_without_helmet"
quedan en train y ninguna en test, no podés evaluar esa clase.

**Cómo lo hicimos:** agrupamos imágenes por la **combinación de clases
presentes** (ej. "tiene casco + chaleco", "tiene solo casco", etc.) y
repartimos cada bucket por separado.

---

### 4.3 Data leakage (fuga de datos)

**Qué es:** cuando información del test se "filtra" al train, haciendo
que el modelo aprenda cosas que no debería conocer.

**Ejemplo típico:** el mismo trabajador aparece en train y en test con
fotos distintas. El modelo "memoriza" la cara del trabajador y predice
bien sin entender el problema.

**Cómo lo evitamos:** una imagen pertenece **exactamente a un split**,
nunca aparece en dos splits.

---

### 4.4 Overfitting (sobreajuste)

**Qué es:** el modelo "memoriza" el training set pero falla en datos
nuevos.

**Cómo detectarlo:** la diferencia val - test grande, o train loss
muy baja con val loss alta.

**Nuestro modelo:** Δ entre val y test = −0.4 pp. **Sin overfitting.**

---

### 4.5 Class imbalance (desbalance de clases)

**Qué es:** algunas clases tienen muchísimas más instancias que otras.

**Nuestro caso:** `person` con 1.116 train vs `head_with_helmet` con
32.055. Hay 28× más cascos que personas.

**Efecto:** el modelo aprende muy bien lo común y muy mal lo raro.
**Nuestro `person` cayó a mAP@0.5 = 0.176 por esto.**

---

## 5. Conceptos de entrenamiento

### 5.1 Epochs

**Qué son:** cada "pasada" completa sobre todas las imágenes de train.

**Nuestro caso:** 80 epochs. Cada epoch toma ~5 minutos en T4.

---

### 5.2 Batch size

**Qué es:** cuántas imágenes ve el modelo a la vez antes de actualizar
sus pesos.

**Nuestro caso:** 16. Limitado por la VRAM de la GPU T4.

---

### 5.3 Learning rate

**Qué es:** "qué tan grande" es cada paso de aprendizaje. Demasiado alto
→ no converge. Demasiado bajo → tarda muchísimo.

**Nuestro caso:** 0.01 inicial, decrece exponencialmente a 0.0001.

---

### 5.4 Patience / Early stopping

**Qué es:** si el modelo no mejora en N epochs seguidos, se detiene.

**Nuestro caso:** `patience=20`. **No se activó** porque el modelo siguió
mejorando hasta el último epoch.

---

### 5.5 Optimizador (SGD)

**Qué es:** el algoritmo que actualiza los pesos del modelo. SGD
(Stochastic Gradient Descent) es el clásico, robusto.

**Nuestro caso:** SGD con momentum=0.937, weight_decay=0.0005.

---

## 6. Augmentaciones

Augmentaciones = transformaciones aleatorias a las imágenes durante
entrenamiento para que el modelo vea más variedad sin tener más datos.

### 6.1 Mosaic

**Qué es:** pegar 4 imágenes en una sola. El modelo ve combinaciones
extrañas.

**Trade-off:** ayuda con generalización pero distorsiona escalas (objetos
pequeños se hacen aún más pequeños).

### 6.2 close_mosaic=15

**Qué es:** desactivar mosaic en los últimos 15 epochs. Permite al
modelo "afinar" sobre imágenes en su distribución natural.

**Nuestro modelo D usa esto.** Mejora marginalmente vs baseline.

### 6.3 MixUp

**Qué es:** combinar 2 imágenes con cierta transparencia.

**Nuestro caso:** no en el modelo final (Exp B la probó, no aportó).

### 6.4 Random Erasing

**Qué es:** borrar pedazos aleatorios de la imagen para simular oclusión.

**Nuestro caso:** Exp B lo probó, sin mejora medible.

### 6.5 HSV jitter, flip, translate, scale

Variaciones de iluminación, espejado horizontal, traslaciones y escala.
**Activos por defecto en todos los modelos.**

---

## 7. Preguntas probables del profesor + cómo responder

### "¿Por qué YOLOv8 y no otro?"

> "Por madurez del tooling (Ultralytics), documentación, compatibilidad
> probada con Kaggle, y excelente relación costo-beneficio frente a
> YOLOv9 para un proyecto con plazo definido. YOLOv8s es la variante
> small (~11M parámetros), balance entre capacidad y tiempo de entrenamiento."

### "¿Qué es mAP@0.5?"

> "Mean Average Precision considerando como acierto todo bbox con IoU
> ≥ 0.5 respecto al ground truth. Es la métrica estándar de Ultralytics
> y de la mayoría de comparativas en la literatura. Mide qué tan bien
> el modelo detecta y clasifica, promediado sobre clases."

### "¿Por qué la clase `person` tiene mAP@0.5 = 0.18?"

> "Porque tiene solo 1.116 instancias en train, mientras que las otras
> tienen entre 10.000 y 32.000. Es una limitación del dataset combinado:
> los datasets que usamos están centrados en EPP (cabeza y chaleco),
> no anotan exhaustivamente cuerpos completos. Esto está documentado
> en el informe como limitación reconocida del dataset, no del modelo."

### "¿Por qué tantas detecciones INDETERMINADO en el video?"

> "Porque el modelo, entrenado sobre 3 datasets con anotaciones parciales
> no superpuestas, rara vez detecta cabezas y chalecos simultáneamente
> en la misma imagen. El verificador, al no tener evidencia suficiente,
> prefiere emitir INDETERMINADO antes que inventar un veredicto. Esto
> está validado experimentalmente: con un modelo entrenado sobre un
> dataset bien anotado, la tasa de INDETERMINADO cae del 27% al 4%."

### "¿No habrían sido mejores otras métricas, como F1 por clase?"

> "F1 está implícito en mAP — la curva PR de cada clase contiene la
> información de F1 a cada umbral. Reportamos precision y recall por
> clase por separado para que el lector pueda calcular F1 si lo
> necesita y para hacer visible el trade-off entre ambas."

### "¿Por qué no usaron Cross-Validation?"

> "Tres razones: (1) Cross-validation implica entrenar el modelo K veces,
> lo que sale fuera del presupuesto de cómputo de Kaggle (30h
> semanales). (2) Para datasets de visión grandes (25.976 imágenes),
> un split honesto train/val/test 70/15/15 estratificado da una
> estimación de generalización suficientemente robusta. (3) Lo
> validamos viendo que val y test dieron métricas casi idénticas
> (−0.4 pp), confirmando que el modelo generaliza."

### "¿Por qué el verificador y no entrenar un clasificador directo CONFORME/NO?"

> "Porque el enunciado lo pide explícitamente y porque es la práctica
> estándar en la industria. La separación percepción/decisión favorece
> auditabilidad (sabés POR QUÉ es no conforme), modificabilidad
> (cambiás reglas sin reentrenar) y métricas detalladas (mAP por clase
> vs solo accuracy binario). Además permite que el modelo se reutilice
> para otras reglas de cumplimiento sin reentrenar."

### "¿Y si hay oclusión en una obra real?"

> "Lo analizamos en la sección 4.4. Sobre el test set, el modelo es
> robusto a oclusión baja-media (mAP@0.5 ≈ 0.78 en ambos). Cae 7.5 pp
> a oclusión alta (mAP@0.5 = 0.70). Lo más importante: el recall cae
> 17.3 pp mientras la precision se mantiene, lo que significa que el
> modelo PIERDE detecciones (no las inventa). En seguridad ocupacional
> ese es el comportamiento más conservador."

### "¿Por qué no entrenaron también con datos sintéticos?"

> "El enunciado lo descartó explícitamente. Y por buena razón: datos
> sintéticos pueden introducir distribuciones distintas a las reales
> y dar falsa sensación de generalización. Para producción valdría la
> pena evaluarlo, pero como hallazgo este parcial documenta primero
> el comportamiento sobre datos reales."

---

## 8. Lo más importante para llevarte a la presentación

| Concepto | Frase clave |
|---|---|
| **mAP@0.5 = 0.771** | "Métrica estándar del campo. Mejor sería 0.85+, pero 0.77 es sólido para baseline académico." |
| **Generalización OK** | "Test = val, sin overfitting. Los splits sin fuga funcionaron." |
| **Person escasa** | "Limitación del dataset, no del modelo. Sin más datos no se arregla con hiperparámetros." |
| **INDETERMINADO** | "Decisión de diseño: el sistema prefiere admitir incertidumbre antes que inventar." |
| **Validación con OtroEnfoque** | "Confirmamos cuantitativamente que el cuello de botella es el dataset combinado, no nuestro pipeline." |
| **Verificador determinista** | "Separación percepción/decisión: auditable, modificable, ético." |

Repasalo el día anterior. Si te trabás con alguna métrica, mirá la
sección 7 (Q&A) y ensayá la respuesta en voz alta una vez.

Mucha suerte 💪
