# Informe en LaTeX (formato IEEE) para Overleaf

Esta carpeta contiene la versión LaTeX del informe final del parcial,
escrita con la clase **IEEEtran** (estilo IEEE Conference, 2 columnas).

## Contenido

```
reports/latex/
├── informe_final.tex     ← documento principal (IEEEtran conference)
├── figures/              ← 8 imágenes referenciadas
│   ├── class_distribution.png
│   ├── results_baseline.png
│   ├── results_exp_D.png
│   ├── confusion_baseline.png
│   ├── confusion_exp_D.png
│   ├── verifier_imagen.jpg
│   ├── demo_modelo_D.jpg
│   └── demo_externo.jpg
└── README.md             ← este archivo
```

## Subir a Overleaf

### Opción A: ZIP (recomendada, 1 paso)

1. Click derecho sobre la carpeta `latex/` (en
   `reports/`) → "Enviar a" → "Carpeta comprimida".
2. En Overleaf: https://www.overleaf.com/project → **New Project**
   → **Upload Project** → seleccioná el `latex.zip`.
3. Overleaf descomprime y abre el proyecto automáticamente.

> Verificá que el `.zip` contenga **directamente** `informe_final.tex`
> y la carpeta `figures/` en su raíz (no anidados dentro de otra
> carpeta `latex/`).

### Opción B: Manual

1. En Overleaf: **New Project** → **Blank Project**.
2. Eliminá el `main.tex` que viene por defecto.
3. Click en el ícono de subir archivo (📤) y subí `informe_final.tex`
   a la raíz.
4. Crear carpeta `figures/` (ícono de carpeta nueva).
5. Entrar a `figures/` y subir las 8 imágenes ahí.

## Compilar en Overleaf

1. Verificá el compilador en **Menu (☰) → Settings → Compiler**:
   - **pdfLaTeX** funciona perfecto (recomendado).
   - XeLaTeX y LuaLaTeX también compilan este documento.
2. Click **"Recompile"** (botón verde arriba).
3. Primera compilación: ~30-60 segundos.
4. El PDF aparece a la derecha. Páginas estimadas: 6-9 (formato IEEE
   es compacto).

## Personalización rápida

### Cambiar autor

Cerca de la línea 55 del `.tex`:
```latex
\author{
    \IEEEauthorblockN{Jesús Percy Nazario Portilla}
    ...
}
```
Reemplazar el nombre. Si hay más de un autor, agregar más bloques
`\IEEEauthorblockN{...}\IEEEauthorblockA{...}` separados por
`\and`.

### Agregar afiliación (universidad, departamento)

Dentro de `\IEEEauthorblockA{...}` agregar las líneas:
```latex
\IEEEauthorblockA{
    \textit{Departamento de Ingeniería de Software} \\
    \textit{Universidad XYZ} \\
    Ciudad, País \\
    correo@universidad.edu
}
```

### Cambiar título del documento

Línea ~50:
```latex
\title{Detección de Cumplimiento de Normas de Seguridad...}
```

### Versión journal en vez de conference

Cambiar la primera línea:
```latex
\documentclass[conference]{IEEEtran}
```
por:
```latex
\documentclass[journal]{IEEEtran}
```
La versión `journal` tiene running header con el título.

## Por qué IEEEtran

- **Estándar internacional**: la mayoría de revistas y conferencias
  académicas de ingeniería usan este formato.
- **Compacto**: 2 columnas, fuentes optimizadas. Documento de 15-20
  páginas en formato article se reduce a 6-9 en IEEE.
- **Profesional**: el estilo es reconocido y respetado.
- **Disponible en Overleaf**: el paquete `IEEEtran` viene preinstalado.

## Estructura del documento

El informe está organizado siguiendo las convenciones IEEE:

1. **Title + Author block** con info del repositorio.
2. **Abstract** + **IEEE Keywords**.
3. **Introducción** (problema, estrategia).
4. **Datos** (fuentes, taxonomía, splits).
5. **Metodología** (modelo, hiperparámetros, experimentos).
6. **Resultados cuantitativos** (tablas + figuras de curvas, matriz).
7. **Verificador determinista** (diseño, lógica, hallazgo del sesgo).
8. **Demostración en video** (con comparación de modelo externo).
9. **Implicaciones éticas** (3 ejes).
10. **Limitaciones y trabajo futuro**.
11. **Conclusiones**.
12. **Bibliografía** (6 referencias).

## Figuras y tablas

Algunas figuras y tablas usan el ambiente con asterisco (`figure*`,
`table*`) para ocupar las **2 columnas** (necesario para gráficos
anchos como las curvas de entrenamiento y la comparativa de modelos).

Las demás figuras usan ancho de columna (`\columnwidth`).

## Errores comunes y soluciones

| Error | Solución |
|---|---|
| `File 'figures/xxx.png' not found` | Verificar que la carpeta `figures/` está al lado del `.tex`, no anidada |
| `Undefined control sequence \IEEEauthorblockN` | Asegurate de tener `\documentclass{IEEEtran}` (no `article`) |
| Caracteres raros (á, é, ñ) | Overleaf guarda en UTF-8 por defecto; verificá en Menu → Settings → Spell check (Spanish) |
| Figuras quedan en otra página | Es normal en IEEE: usar `[!h]` o `[!t]` para forzar posición, o aceptar el flujo automático |

## Referencias bibliográficas

El documento incluye 6 referencias (YOLO, Ultralytics, Roboflow, SHWD,
Supervision, COCO). Si necesitás agregar más:

```latex
\bibitem{tu_referencia}
Autor, ``Título,'' Revista, año.
```

Y citalas en el texto con `\cite{tu_referencia}`.
