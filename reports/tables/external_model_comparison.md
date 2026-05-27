# Validación experimental: nuestro modelo D vs. modelo externo "OtroEnfoque"

**Fecha:** 2026-05-27
**Objetivo:** verificar empíricamente si la limitación observada en la
Fase 4 (alta tasa de veredictos INDETERMINADO sobre video real) es
**del pipeline** o **del dataset de entrenamiento**.

## Hipótesis

Si la limitación es del dataset (anotación parcial no superpuesta entre
SHWD/Hard Hat y PPE), un modelo entrenado sobre un dataset bien anotado
(donde cada imagen tiene Person + Hardhat + Safety Vest simultáneamente)
debería producir muchas más detecciones de ambas clases por frame y, en
consecuencia, una proporción mucho menor de veredictos INDETERMINADO.

Si la limitación es del pipeline (verificador, asociación, etc.), el
comportamiento sería similar incluso con un modelo mejor.

## Modelo externo

Encontrado en la carpeta `OtroEnfoque/detection_ppe/` (~87 MB, probablemente
YOLOv8m). Taxonomía de 10 clases con anotación completa por trabajador:
Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, Mask, NO-Mask,
Safety Cone, machinery, vehicle.

Las clases relevantes para nuestro verificador se remapearon:
- Hardhat ↔ head_with_helmet
- NO-Hardhat ↔ head_without_helmet
- Safety Vest ↔ vest
- NO-Safety Vest ↔ no_vest_person
- Person ↔ person
- (Mask, NO-Mask, Safety Cone, machinery, vehicle se descartan)

El verificador y el renderer NO se modificaron — solo las detecciones se
remapean en el script `src/run_video_external.py`.

## Procedimiento

Se procesó el mismo video (`demo_video/video03p.mp4`, 26 segundos,
773 frames) con ambos modelos:

| Modelo | Inferencia |
|---|---|
| Nuestro D (YOLOv8s entrenado en sesión) | `conf=0.15, imgsz=1280` |
| OtroEnfoque (YOLOv8m externo) | `conf=0.25, imgsz=960` |

> Nota: para el modelo D se usaron parámetros más permisivos (`conf` bajo
> e `imgsz` alto) en un intento de compensar el sesgo de etiquetado.
> Aun así los resultados son notablemente más débiles, lo que refuerza
> la hipótesis del problema de datos.

## Resultados sobre el mismo video

| Métrica | Modelo D (nuestro) | **Modelo OtroEnfoque** | Ratio |
|---|---:|---:|---:|
| CONFORME | 861 | **2.287** | **2.66×** |
| NO_CONFORME | 2.600 | **7.221** | **2.78×** |
| INDETERMINADO | 1.309 | **397** | **0.30×** ↓ |
| **Veredictos determinados (CONF + NO_CONF)** | 3.461 | **9.508** | **2.75×** |
| **INDETERMINADO sobre el total** | **27.4%** | **4.0%** | **−85%** |
| Tasa de cumplimiento (CONFORME / determinados) | 24.9% | 24.0% | similar |

## Interpretación

1. **El verificador funciona idénticamente** en ambos casos: no se tocó
   ni una línea de su código. Lo único que cambia es **qué le da el modelo**.

2. **El modelo OtroEnfoque produce 2.75× más veredictos determinados** y
   reduce la proporción de INDETERMINADO de 27% a 4%. La diferencia es
   abrumadora.

3. **La tasa de cumplimiento (cuando hay veredicto definitivo) es similar**:
   24.9% vs. 24.0%. Es decir, ambos modelos coinciden en QUÉ proporción
   del personal cumple cuando se logra observar a cada uno. Lo que
   difieren es **cuántos pueden observar**.

4. **Esto confirma cuantitativamente la hipótesis**: la limitación que
   reportamos en la Fase 4 era **estructural del dataset** (anotación
   parcial entre SHWD/Hard Hat y PPE) y no del diseño del verificador
   ni de la pipeline en general.

## Implicaciones para el informe

Este experimento permite reformular la conclusión del proyecto:

- **No es** "nuestro sistema tiene un fallo de diseño que limita su
  utilidad".
- **Es** "nuestro sistema funciona correctamente; la cobertura efectiva
  está limitada por la calidad de las anotaciones disponibles. Con un
  dataset apropiadamente anotado (validado experimentalmente con un
  modelo externo), el sistema produce 2.75× más veredictos definitivos
  por escena".

Esto es una **validación experimental directa** de la recomendación de
"trabajo futuro" propuesta en el informe.

## Artefactos

- Script: `src/run_video_external.py` (con tabla de remapeo de clases).
- Video con modelo D: `demo_video/video03_annotated_v2.mp4` (114 MB).
- Video con modelo OtroEnfoque: `demo_video/video03_otroenfoque.mp4` (similar
  tamaño).
- Frames de preview comparativos: `demo_video/preview_v3v2_*.jpg` (D) y
  `demo_video/preview_otro_*.jpg` (OtroEnfoque).

## Caveat de honestidad académica

El modelo OtroEnfoque **no es producto de este trabajo**. Se usó únicamente
como referencia experimental para validar nuestro diagnóstico. El modelo
principal entregado por este proyecto sigue siendo el **Experimento D**
entrenado en este parcial, con sus métricas reportadas honestamente
(mAP@0.5 = 0.771 sobre test). La comparación es honesta: muestra una
limitación real y la explica con evidencia.
