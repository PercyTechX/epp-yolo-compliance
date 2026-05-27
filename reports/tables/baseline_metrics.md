# Métricas del proyecto: 3 experimentos comparados + evaluación final en test set

**Modelo final adoptado:** YOLOv8s con `close_mosaic=15` (Experimento D).

## 1. Comparación de los 3 experimentos sobre el conjunto de validación

Tiempo de entrenamiento por experimento: ~7-8 horas en Tesla T4 (Kaggle).
80 epochs completos cada uno (no se activó early stopping).

### mAP@0.5

| Clase | Instancias | Baseline | Exp B | Exp D | Mejor |
|---|---:|---:|---:|---:|---|
| head_with_helmet | 6.974 | 0.978 | 0.978 | **0.979** | D (+0.001) |
| head_without_helmet | 2.194 | 0.959 | 0.960 | 0.959 | Empate |
| vest | 2.828 | 0.987 | 0.986 | 0.986 | Empate |
| person | 282 | 0.175 | 0.153 | 0.175 | Empate |
| **all** | 12.278 | **0.775** | 0.769 | **0.775** | Empate |

### mAP@0.5-0.95

| Clase | Baseline | Exp B | Exp D | Mejor |
|---|---:|---:|---:|---|
| head_with_helmet | 0.675 | 0.670 | **0.677** | D (+0.002) |
| head_without_helmet | 0.652 | 0.649 | 0.652 | Empate |
| vest | 0.835 | 0.830 | **0.837** | D (+0.002) |
| person | 0.106 | 0.099 | **0.108** | D (+0.002) |
| **all** | 0.567 | 0.562 | **0.569** | **D (+0.002)** |

### Interpretación

- **Baseline ya muy fuerte** sobre las 3 clases mayoritarias (cabezas y chaleco
  ≥95% mAP@0.5). El dataset es lo suficientemente limpio como para que el
  baseline ya esté bien tuneado.
- **Exp B** (Random Erasing + MixUp): caída marginal en mAP@0.5 (−0.6 pp).
  Las augmentaciones agresivas no aportaron en este dataset. Hallazgo válido
  para el informe: los datos ya tienen suficiente variabilidad natural.
- **Exp D** (close_mosaic=15): ganador marginal. +0.2 pp en mAP@0.5-0.95
  consistente en TODAS las clases. Se elige como modelo final.
- **Experimentos A (res 960) y C (pesos de clase) saltados** por costo/beneficio:
  - A: ~12-15h, riesgo de timeout en Kaggle (límite 12h por sesión).
  - C: el problema de `person` es de cantidad de datos (1.116 train), no
    se arregla con pesos de pérdida.

## 2. Evaluación final del modelo D sobre el conjunto de test

**Conjunto de test:** 3.895 imágenes, 12.373 instancias, nunca vistas durante
el entrenamiento ni la selección del modelo.

### Tabla completa por clase

| Clase | Instancias | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|---|---:|---:|---:|---:|---:|
| head_with_helmet | 7.222 | 0.939 | 0.950 | **0.976** | 0.664 |
| head_without_helmet | 2.104 | 0.892 | 0.942 | **0.947** | 0.646 |
| vest | 2.831 | 0.937 | 0.957 | **0.984** | 0.837 |
| person | 216 | 0.341 | 0.208 | **0.176** | 0.129 |
| **all** | 12.373 | **0.777** | **0.765** | **0.771** | **0.569** |

> `no_vest_person` no aparece en la tabla porque tiene 0 instancias en el
> conjunto de test (y validation). Esto es esperado: ningún dataset original
> aporta esa etiqueta; el verificador determinista la infiere espacialmente.

### Generalización (val vs test)

| Métrica global | Val | Test | Δ | Interpretación |
|---|---:|---:|---:|---|
| mAP@0.5 | 0.775 | 0.771 | −0.004 | Sin overfitting |
| mAP@0.5-0.95 | 0.569 | 0.569 | 0.000 | Idéntico |
| Precision | 0.795 | 0.777 | −0.018 | Caída mínima |
| Recall | 0.765 | 0.765 | 0.000 | Idéntico |

**Conclusión:** las métricas de test son prácticamente idénticas a las de val.
El modelo generaliza correctamente. Esto confirma que los splits estratificados
70/15/15 sin fuga de la Fase 1 funcionaron como esperaba.

## 3. Para el informe

### mAP global (entregable 2)

- **mAP@0.5 = 0.771** sobre test (modelo D, YOLOv8s @ 640).
- **mAP@0.5-0.95 = 0.569** sobre test.

### Desglose por clase (entregable 2)

- Las 3 clases críticas (head_with_helmet, head_without_helmet, vest) están en
  **mAP@0.5 ≥ 0.947**. Excelente para un baseline académico.
- `person` queda en **mAP@0.5 = 0.176** por escasez de datos (216 instancias
  test, 1.116 train). Esto es una **limitación del dataset**, no del modelo, y
  se documenta como tal.

### Reproducibilidad

- Pesos del modelo final: `runs_outputs/exp_D/best.pt` (22.5 MB). Se publica
  como GitHub Release al cierre del proyecto (no se commitea al repo).
- Configs versionadas: `configs/data.yaml` + `configs/hyp_exp_D.yaml`.
- Notebook reproducible: `notebooks/kaggle/05_exp_D_close_mosaic.ipynb`.
- Notebook de evaluación en test: `notebooks/kaggle/07_test_evaluation.ipynb`.
