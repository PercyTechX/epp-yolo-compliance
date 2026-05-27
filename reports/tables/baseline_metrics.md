# Métricas del modelo baseline (Fase 2)

**Modelo:** YOLOv8s
**Hiperparámetros:** `configs/hyp_baseline.yaml`
**Resolución:** 640
**Epochs:** 80 completos (sin early stopping)
**Tiempo de entrenamiento:** 7h 19m en Tesla T4 (Kaggle)
**Conjunto de validación:** 3.897 imágenes, 12.278 instancias

## Métricas por clase

| Clase | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 | Instancias |
|---|---:|---:|---:|---:|---:|
| head_with_helmet | 0.943 | 0.955 | 0.978 | 0.675 | 6.974 |
| head_without_helmet | 0.923 | 0.945 | 0.959 | 0.652 | 2.194 |
| vest | 0.939 | 0.971 | 0.987 | 0.835 | 2.828 |
| person | 0.426 | 0.174 | 0.175 | 0.106 | 282 |
| no_vest_person | — | — | — | — | 0 (no presente) |
| **all (promedio)** | **0.808** | **0.761** | **0.775** | **0.567** | 12.278 |

## Falsos negativos (instancias perdidas) según matriz de confusión

| Clase | Perdidas / total | % perdido |
|---|---|---:|
| head_with_helmet | 216 / 6.953 | 3.1% |
| head_without_helmet | 78 / 2.174 | 3.6% |
| vest | 74 / 2.828 | 2.6% |
| person | 227 / 282 | **80.5%** |

## Observaciones cualitativas

- **Sin confusión entre clases**: el modelo distingue casco vs. no-casco con
  solo 20 errores cruzados sobre ~7.000 instancias. Excelente separabilidad.
- **`person` no aprendida**: 80% de instancias perdidas, totalmente atribuible
  al desbalance de datos (1.116 train vs. >10k de cabezas). Limitación
  documentada del dataset, no del modelo.
- **Robustez para el verificador**: las 3 clases que el verificador necesita
  (head_with_helmet, head_without_helmet, vest) están en ≥95% mAP@0.5.
  El módulo de cumplimiento puede operar con confianza sobre estas.

## Para el informe final

- **mAP@0.5 global = 0.775** (con `person` arrastrando).
- **mAP@0.5 ponderado excluyendo `person` ≈ 0.97** (estimado).
- Reportar AMBOS números, explicando la limitación de la clase `person`.

## Artefacto

Pesos finales: `runs/baseline/weights/best.pt` (22.5 MB), descargado a la
raíz del proyecto. No commitear al repo; va a un release de Git al final.
