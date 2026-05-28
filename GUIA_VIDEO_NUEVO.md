# Guía rápida: procesar un video nuevo

Esta guía explica cómo correr el sistema completo (YOLO + verificador) sobre
**un video que te dé el profesor** o cualquier video nuevo.

---

## Opción A: Arrastrar y soltar (lo más fácil) ⭐

1. Pedile al profesor que te pase el video. Guardalo en cualquier carpeta de
   tu computadora.
2. Abrí el Explorador de Windows.
3. Andá a: `C:\Users\Percy Tech\Desktop\PARCIAL DEEP LEARNING\`
4. **Arrastrá el archivo de video** desde donde esté hasta el archivo
   **`procesar_video.bat`** (sin soltarlo, ponelo encima del .bat).
5. Soltá. Se abre una ventana negra (PowerShell o cmd) que procesa el video.
6. Esperá entre 30 segundos y 5 minutos (depende del largo del video).
7. Cuando termine, vas a ver el mensaje "OK. Video anotado guardado en..."
   y la ruta del archivo de salida.

El video procesado queda en: `demo_video\<nombre_original>_annotated.mp4`

---

## Opción B: Por terminal (más control)

### Paso 1: Guardar el video

Guardá el video del profesor en la carpeta `demo_video\` con un nombre simple
sin espacios, por ejemplo:
```
C:\Users\Percy Tech\Desktop\PARCIAL DEEP LEARNING\demo_video\video_profesor.mp4
```

### Paso 2: Abrir terminal en la carpeta del proyecto

1. En el Explorador, andá a:
   `C:\Users\Percy Tech\Desktop\PARCIAL DEEP LEARNING\`
2. **En la barra de dirección de arriba**, borrá lo que dice y escribí:
   ```
   powershell
   ```
   Apretá Enter. Se abre PowerShell directamente en esa carpeta.

> Alternativa: shift + click derecho dentro de la carpeta → "Abrir ventana
> de PowerShell aquí".

### Paso 3: Correr el comando

Copiá y pegá esto en la terminal (ajustando el nombre del video):

```powershell
python -m src.run_video --weights runs_outputs/exp_D/best.pt --video demo_video/video_profesor.mp4 --out demo_video/video_profesor_annotated.mp4 --conf 0.15 --imgsz 1280
```

**Reemplazá `video_profesor.mp4`** por el nombre real del archivo.

Apretá Enter y esperá. Vas a ver el progreso:
```
[video] 30/500 frames (3s)
[video] 60/500 frames (5s)
...
[video] OK. Video anotado: demo_video\video_profesor_annotated.mp4
[video] Veredictos: {'CONFORME': 123, 'NO_CONFORME': 45, 'INDETERMINADO': 67}
```

---

## Parámetros que podés ajustar

| Parámetro | Default | Cuándo cambiarlo |
|---|---|---|
| `--conf 0.15` | 0.25 | Si el modelo "no ve nada", bajar a 0.10. Si ves muchos falsos positivos, subir a 0.30 |
| `--imgsz 1280` | 640 | Trabajadores chicos en el video → subir a 1280 o 1600. Trabajadores grandes → bajar a 640 |
| `--frame-skip 1` | 1 | Si el video es muy largo, poné 2 o 3 para acelerar (procesa 1 de cada N frames) |

Ejemplos:

**Video con trabajadores muy lejos (objetos pequeños):**
```powershell
python -m src.run_video --weights runs_outputs/exp_D/best.pt --video demo_video/video.mp4 --out demo_video/salida.mp4 --conf 0.15 --imgsz 1600
```

**Video largo (5+ minutos):**
```powershell
python -m src.run_video --weights runs_outputs/exp_D/best.pt --video demo_video/video.mp4 --out demo_video/salida.mp4 --conf 0.15 --imgsz 1280 --frame-skip 3
```

---

## Qué tiene el video de salida

- **Bbox verdes** = CONFORME (trabajador con casco + chaleco detectados).
- **Bbox rojos** = NO_CONFORME (con razón impresa: `casco_ausente`, `chaleco_no_visible`).
- **Bbox amarillos** = INDETERMINADO (chaleco sin cabeza asociada, o ambiguo).
- **Esquina superior izquierda:** "Cumplimiento: XX%" — tasa por fotograma.

---

## Cómo verlo

Abrí el archivo `_annotated.mp4` resultante con:
- VLC Media Player (recomendado, abre cualquier MP4 sin problemas)
- Windows Media Player (puede no abrir algunos códecs)
- O subílo a Google Drive y miralo desde ahí

---

## Errores comunes y soluciones

### `ModuleNotFoundError: No module named 'src'`

**Causa:** estás corriendo el comando desde otra carpeta.

**Solución:** asegurate que la terminal está en
`C:\Users\Percy Tech\Desktop\PARCIAL DEEP LEARNING\`. Para verificar,
escribí `pwd` y apretá Enter — debería mostrar esa ruta.

### `FileNotFoundError: best.pt`

**Causa:** el modelo no está donde el script lo busca.

**Solución:** verificá que existe `runs_outputs\exp_D\best.pt`. Si no,
descargalo del Release de GitHub:
https://github.com/PercyTechX/epp-yolo-compliance/releases/tag/v1.0

### `cv2 no encontrado` o `ultralytics no encontrado`

**Causa:** las dependencias no están instaladas.

**Solución:** correr en la terminal:
```powershell
python -m pip install ultralytics opencv-python supervision
```

### El video procesa pero queda con todo INDETERMINADO

**Causa:** el modelo no detecta cabezas en ese video (cámara muy lejos, o
estilo de imagen muy diferente al training set).

**Solución:** probar con `--imgsz 1600` y `--conf 0.10`. Si aun así no
mejora, es la limitación del dataset documentada en el informe (sección 5.4).

### El video sale al revés (de cabeza)

**Causa:** algunos videos verticales de celular tienen metadata de rotación
que OpenCV no respeta.

**Solución:** abrir el video original con VLC, ir a "Herramientas →
Efectos y filtros → Vídeo → Geometría → Rotar 180°", guardarlo así
y volver a procesarlo.

---

## En la presentación del parcial

Si el profesor te da un video en ese momento:

1. **Conectá una USB o pasalo por la nube** al directorio
   `demo_video\`.
2. **Arrastrá al `procesar_video.bat`** o usá el comando de la Opción B.
3. Mientras procesa, explicá lo que está haciendo:
   - "Esto carga el modelo YOLOv8s que entrené (22 MB)."
   - "Para cada fotograma del video corre detección + verificador."
   - "El verificador asigna verde/rojo/amarillo según la lógica de reglas."
4. Cuando termine, abrí el `_annotated.mp4` con VLC.
5. Mostrá los veredictos en tiempo real.

**Tiempo total**: el script tarda ~1-2 segundos por frame en CPU, así que
para un video de 30 segundos a 30 fps (900 frames), unos 15-30 minutos. Si
te apura el tiempo, usá `--frame-skip 3` y baja a ~5-10 minutos.

> Tip: pre-procesá los videos que te pase el profesor antes de la
> presentación, y mostralos directamente. Si te pide hacerlo en vivo,
> arrancá el procesamiento al inicio de la presentación y mientras
> explicás otras partes del informe va corriendo en background.
