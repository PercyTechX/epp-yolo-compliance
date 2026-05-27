# Análisis de robustez por nivel de oclusión (Fase 5.2 — entregable 3)

**Modelo evaluado:** `runs_outputs/exp_D/best.pt` (YOLOv8s con `close_mosaic=15`).
**Conjunto evaluado:** test set (3.895 imágenes nunca vistas).
**Procedimiento:** para cada imagen, se calcula el IoU máximo entre cualquier
par de cajas anotadas. Ese IoU clasifica la imagen en un nivel de oclusión.

## Umbrales

| Nivel | IoU máximo entre cajas |
|---|---|
| Low | `< 0.15` |
| Medium | `[0.15, 0.40)` |
| High | `≥ 0.40` |

## Distribución de imágenes por nivel

| Nivel | Imágenes | % del test set |
|---|---:|---:|
| Low | 3.695 | 94.9% |
| Medium | 185 | 4.7% |
| High | 15 | 0.4% |
| **Total** | **3.895** | **100%** |

La distribución refleja el sesgo del dataset compuesto: la mayoría de las
imágenes son trabajadores aislados o posados en planos limpios. Escenas
densamente ocluidas son raras en SHWD/Hard Hat/PPE.

## Resultados por nivel

| Nivel | n | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Low | 3.695 | **0.779** | 0.576 | 0.782 | **0.796** |
| Medium | 185 | 0.789 | 0.565 | 0.797 | 0.749 |
| High | 15 | **0.704** | 0.512 | 0.783 | **0.623** |

### Caída de low a high

| Métrica | Δ (high − low) |
|---|---:|
| mAP@0.5 | **−7.5 pp** |
| mAP@0.5-0.95 | −6.4 pp |
| Precision | +0.1 pp (sin cambio relevante) |
| Recall | **−17.3 pp** |

## Interpretación para el informe

1. **El modelo es robusto a oclusión leve-media**: las métricas en low y medium
   son estadísticamente equivalentes (mAP@0.5 ≈ 0.78 en ambos). No se observa
   degradación medible en el rango bajo-medio.

2. **Oclusión alta sí degrada el desempeño**: caída de 7.5 pp en mAP@0.5 y de
   6.4 pp en mAP@0.5-0.95 contra el nivel low. La degradación es consistente
   con lo esperado: cuando varios trabajadores se solapan, el modelo tiende a
   no separarlos bien.

3. **El recall es la métrica más afectada (−17.3 pp)**, no la precisión
   (que se mantiene estable en ~0.78). Esto significa que el modelo **pierde
   detecciones** (no las ve) cuando hay oclusión fuerte, pero las pocas que
   encuentra siguen bien clasificadas. Operativamente, esto es importante
   para el verificador: en escenas concurridas se podrían perder
   trabajadores no conformes, lo que es un falso negativo crítico
   (riesgo de seguridad pasado por alto).

4. **Caveat estadístico**: el grupo "high" solo tiene 15 imágenes (162
   instancias). El estimador de mAP tiene varianza alta y el intervalo de
   confianza del "−7.5 pp" es amplio. Sería deseable validar este hallazgo
   con un dataset que incluya más escenas densas.

## Implicaciones de diseño

- **Para el verificador**: en escenas con oclusión alta cabe esperar más
  veredictos INDETERMINADO (cabeza o chaleco no detectado), no más
  falsos NO_CONFORME. La asimetría precision/recall observada lo respalda:
  no inventa detecciones espurias, simplemente le faltan algunas.
- **Para futuras mejoras**: el experimento B (Random Erasing + MixUp)
  intentó atacar este eje y no produjo mejora en el dataset completo.
  Podría revisitarse con un dataset que tenga mayor proporción de escenas
  ocluidas en el split de entrenamiento.

## Reproducibilidad

- Script: `src/analysis/occlusion.py` (10 tests unitarios pasando).
- Notebook: `notebooks/kaggle/08_occlusion_analysis.ipynb`.
- JSON crudo: `/kaggle/working/occlusion_results.json` del notebook ejecutado.
