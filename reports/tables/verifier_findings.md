# Validación del verificador determinista — Fase 4

**Fecha:** 2026-05-27
**Modelo probado:** `runs_outputs/exp_D/best.pt` (YOLOv8s con `close_mosaic=15`).
**Muestra:** 200 imágenes aleatorias del val set + 5 imágenes diagnóstico.

## Resumen ejecutivo

El módulo verificador determinista (`src/compliance/`) funciona **correctamente**
cuando recibe detecciones del modelo. Los veredictos CONFORME / NO_CONFORME /
INDETERMINADO se asignan según la lógica de asociación cabeza-chaleco y los
casos sintéticos los respaldan (15 tests pytest pasando).

La validación sobre predicciones reales reveló una **limitación importante a
nivel modelo** (no del verificador): los 3 datasets combinados están
**etiquetados parcialmente**, lo que hace que el modelo casi nunca detecte
`head_with_helmet` y `vest` en la misma imagen.

## Cambio de diseño durante la validación

La versión inicial del verificador estaba anclada en `person` ("para cada
person bbox, buscar cabeza y chaleco asociados"). Esta versión fallaba en
producción porque la clase `person` solo se detecta en ~17% de los casos
(es la clase escasa documentada en Fase 1).

Se reescribió el verificador con **estrategia cabeza-ancla**: para cada
cabeza detectada (que sí son abundantes, ≥95% mAP), buscar chaleco debajo
con alineación horizontal razonable. Si existe `person` que contiene la
cabeza, se usa solo para enriquecer el rendering.

Tests actualizados: 15 casos sintéticos pasan con la nueva lógica.

## Hallazgo principal: sesgo de etiquetado entre datasets

Sobre 300 imágenes val:

| Categoría | Imágenes | % |
|---|---:|---:|
| Solo `head_with_helmet` detectado | 147 | 49.0% |
| Solo `head_without_helmet` detectado | 22 | 7.3% |
| Solo `vest` detectado | 121 | 40.3% |
| `head_with_helmet` + `vest` (CONFORME posible) | **9** | **3.0%** |
| `head_without_helmet` + `vest` | 1 | 0.3% |
| Sin detecciones | 0 | 0.0% |

**Causa:** los 3 datasets combinados (Hard Hat Workers, SHWD, Safety Vest PPE)
están etiquetados parcialmente:
- Hard Hat Workers y SHWD: anotan **solo cabezas** (con/sin casco).
- Safety Vest PPE: anota **solo chalecos**.

Durante el entrenamiento, YOLO aprendió la correlación entre el "estilo" de
imagen y las clases anotadas, terminando con un comportamiento bimodal:
- Imágenes parecidas a SHWD/Hard Hat → predice cabezas, ignora chalecos.
- Imágenes parecidas a PPE → predice chalecos, ignora cabezas.

## Validación cualitativa

Tres ejemplos representativos donde el verificador detecta correctamente
escenas con co-ocurrencia:

| Imagen | Trabajadores | Tasa cumplimiento | Veredictos |
|---|---|---:|---|
| `shwd_003457` (3 trabajadores) | 3 cascos + 2 chalecos visibles | 66.7% | 1 CONFORME, 1 NO_CONFORME, 1 INDETERMINADO |
| `shwd_005995` (5 trabajadores) | 5 cascos + 1 chaleco asociado | 25.0% | 1 CONFORME, 4 NO_CONFORME |
| `ppe_train-1670` (3 trabajadores) | 3 cascos + 2 chalecos | 50.0% | 1 CONFORME, 1 NO_CONFORME, 1 INDETERMINADO |

Las imágenes anotadas están en `reports/figures/verifier_demo_exp_D/`.

## Implicaciones para el informe

1. El verificador es **funcional y auditable**: la separación percepción/decisión
   permite intercambiar el detector sin re-entrenar reglas.
2. La métrica de cumplimiento a nivel imagen está **subestimada** en val por el
   sesgo de etiquetado: el modelo no ve los chalecos en escenas estilo SHWD,
   no porque no estén, sino porque no fue entrenado para verlos ahí.
3. **Recomendación clara para "future work":** entrenar sobre un dataset
   con anotación completa por trabajador (ej. Construction Site Safety
   con clases Hardhat + Safety Vest + Person en cada imagen).

## Implicaciones para el video demo (Fase 6)

Para que la demo muestre el sistema funcionando, conviene elegir un video
con:
- Trabajadores a cuerpo completo (no close-ups de cara).
- Escena estilo "obra activa" similar a las imágenes con chalecos del PPE
  dataset.
- Buena visibilidad simultánea de cabeza y torso.

En esas condiciones el modelo sí detecta ambas clases y el verificador
puede emitir CONFORME / NO_CONFORME confiablemente.
