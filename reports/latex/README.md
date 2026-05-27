# Informe en LaTeX para Overleaf

Esta carpeta contiene la versión LaTeX del informe final del parcial.

## Contenido

```
reports/latex/
├── informe_final.tex     ← documento principal LaTeX
├── figures/              ← imágenes referenciadas
│   ├── class_distribution.png        (Distribución de clases por split)
│   ├── results_baseline.png          (Curvas de entrenamiento del baseline)
│   ├── results_exp_D.png             (Curvas de entrenamiento del modelo D)
│   ├── confusion_baseline.png        (Matriz de confusión del baseline)
│   ├── confusion_exp_D.png           (Matriz de confusión del modelo D)
│   ├── verifier_imagen.jpg           (Ejemplo del verificador en val)
│   ├── demo_modelo_D.jpg             (Frame del video demo - modelo D)
│   └── demo_externo.jpg              (Frame del video demo - modelo externo)
└── README.md             ← este archivo
```

## Subir a Overleaf

### Opción A: Crear proyecto vacío y subir manualmente

1. Andá a https://www.overleaf.com/project
2. Click **"New Project"** → **"Blank Project"**.
3. Ponele de nombre: `Parcial Deep Learning EPP YOLO`.
4. Una vez abierto, **eliminá** el `main.tex` que viene por defecto.
5. **Subí los archivos**:
   - Click en el ícono de subir (📤) arriba a la izquierda en el panel de archivos.
   - Subí `informe_final.tex` a la raíz del proyecto.
   - Subí cada archivo de `figures/` adentro de una carpeta `figures/` en Overleaf. Para crear la carpeta, click en el ícono de nueva carpeta antes de subir.

### Opción B: Subir como ZIP (más fácil)

1. Comprimí la carpeta `reports/latex/` entera (click derecho → Enviar a → Carpeta comprimida).
2. En Overleaf: **New Project** → **Upload Project** → seleccioná el `.zip`.
3. Overleaf descomprime y arma todo automáticamente.

> Si subís por ZIP, asegurate que el `.zip` contenga directamente
> `informe_final.tex` y la carpeta `figures/` en su raíz, no anidados.

## Compilar

1. Una vez en Overleaf, click en **"Recompile"** (botón verde arriba).
2. La primera compilación tarda 30-60 segundos.
3. Si todo está bien, vas a ver el PDF a la derecha.

### Compilador recomendado

- Engine: **pdfLaTeX** (default en Overleaf, debería funcionar).
- Si querés usar otro: XeLaTeX o LuaLaTeX también compilan este documento.

### Si hay errores

Errores comunes y soluciones:

| Error | Causa | Solución |
|---|---|---|
| `! LaTeX Error: File 'figures/xxx.png' not found` | Imagen no subida o mal ubicada | Verificar que las imágenes están en la carpeta `figures/` |
| `! Undefined control sequence` por `\url` | Falta paquete | Ya está incluido `\usepackage{hyperref}`, no debería pasar |
| Caracteres raros (`á`, `é`) se ven mal | Encoding mal | Asegurate que el `.tex` se subió como UTF-8 (default de Overleaf) |

## Personalización rápida

### Cambiar autor

Línea ~78 del `.tex`:
```latex
\author{Jesús Percy Nazario Portilla}
```
Cambialo a tu nombre completo.

### Cambiar fecha

Línea ~79:
```latex
\date{\today}
```
- `\today` = fecha de compilación automática.
- O hardcodealo: `\date{27 de mayo de 2026}`.

### Quitar la sección de validación con modelo externo

Si por alguna razón no querés mencionar OtroEnfoque, eliminá la
subsección 6.4 (`\subsection{Validación experimental complementaria}`)
y todo su contenido hasta el comentario `Aclaración de honestidad
académica`.

### Agregar logo de la institución

Antes del `\maketitle`:
```latex
\begin{center}
\includegraphics[width=4cm]{figures/logo_institucion.png}
\end{center}
```
(Tenés que subir el logo a la carpeta `figures/` primero).

## Páginas estimadas

El PDF compilado debería tener entre 15 y 20 páginas incluyendo
figuras y tablas. Si la institución pide un máximo específico,
podés ajustar:

- Reducir tamaño de figuras: `\includegraphics[width=0.7\textwidth]{...}` → `width=0.5\textwidth`.
- Eliminar el apéndice de reproducibilidad si no es necesario.
- Cambiar márgenes: `\usepackage[a4paper,margin=2cm]{geometry}` (de 2.5cm a 2cm).
