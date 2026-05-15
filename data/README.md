# Datos

Esta carpeta **no se versiona** (ver `.gitignore`). Cada integrante debe
poblarla manualmente con los datasets de origen.

## Estructura esperada

```
data/
├── raw/                    # Descargas originales sin modificar
│   ├── shwd/
│   │   ├── images/
│   │   └── labels/
│   ├── helmet/             # Safety Helmet Detection (Roboflow)
│   │   ├── images/
│   │   └── labels/
│   └── ppe/                # PPE Detection (Roboflow)
│       ├── images/
│       └── labels/
│
├── interim/                # Generado por src/data/remap_labels.py
│   ├── shwd/
│   ├── helmet/
│   └── ppe/
│
└── processed/              # Generado por src/data/make_splits.py
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

## Descarga manual

| Dataset | Enlace | Notas |
|---|---|---|
| SHWD | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset | Convertir de VOC a YOLO si es necesario antes de remapear. |
| Safety Helmet Detection | https://roboflow.com (Universe) | Exportar en formato YOLOv8. |
| PPE Detection | https://roboflow.com (Universe) | Exportar en formato YOLOv8. |

Verificar la licencia de cada export. Roboflow Universe suele ser CC BY 4.0.

## Formato esperado de cada dataset en `raw/`

Estructura compatible Ultralytics YOLO:

- `images/` con archivos `.jpg`, `.jpeg` o `.png`.
- `labels/` con archivos `.txt` del mismo nombre base que la imagen.
- Cada línea de un `.txt`: `class_id x_center y_center width height`
  (coordenadas normalizadas en `[0, 1]`).

Si el dataset trae splits (train/val/test) ya separados, conviene aplanarlos
a un solo directorio `images/` + `labels/` antes de correr `remap_labels.py`;
el split estratificado de este proyecto se hace de cero en `make_splits.py`.

## Tras remapeo (`data/interim/`)

Cada subcarpeta tiene la misma estructura `images/` + `labels/` pero las
etiquetas usan los ids de la **taxonomía unificada** definida en
`configs/class_mapping.yaml`:

| id | clase                |
|----|----------------------|
| 0  | head_with_helmet     |
| 1  | head_without_helmet  |
| 2  | person               |
| 3  | vest                 |
| 4  | no_vest_person       |

## Tras splits (`data/processed/`)

Estructura final que Ultralytics consume vía
[configs/data.yaml](../configs/data.yaml). Esta carpeta es la que se sube
como Kaggle Dataset (una sola vez) y se referencia desde los notebooks.
