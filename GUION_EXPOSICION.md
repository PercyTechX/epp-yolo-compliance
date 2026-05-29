# Guion de exposición: cómo contar el proyecto paso a paso

Documento narrativo para la defensa oral del parcial. Sigue la
cronología del desarrollo real para que la historia tenga lógica.

---

## Antes de empezar

### Tiempo total de exposición

Asumiendo **15-20 minutos** de exposición + 5 min de preguntas. Si
tenés más o menos tiempo, ajustá pasando por todas las secciones
pero gastando más o menos en cada una.

### Qué llevar

1. **Laptop** con el proyecto cargado y todas las dependencias.
2. **Slides** (PowerPoint, Google Slides, o solo el PDF del informe IEEE).
3. **Backup**: el video ya procesado por si en vivo falla algo
   (`demo_video/video03_annotated_v2.mp4`).
4. **VLC instalado** para reproducir el video con seguridad.
5. **Conexión a internet** opcional (para mostrar el repo de GitHub).
6. **Este guion impreso** o en el celular para apoyo visual.

### Qué mostrar en pantalla durante la exposición

| Momento | Pantalla |
|---|---|
| Apertura | Slide con título + tu nombre |
| Problema y estrategia | Slides con diagrama de pipeline |
| Datos | Slide con las 3 fuentes + tabla de taxonomía |
| Resultados | Slide con la matriz de confusión + tabla de mAP |
| Demo | Reproducir el video anotado |
| Conclusiones | Slide con las 6 conclusiones |

---

## 📖 Guion

### Sección 1: Apertura (~1 min)

**Qué decir:**

> "Buenos días/tardes. Mi nombre es Percy Nazario y voy a presentar mi
> parcial de Deep Learning, que es un sistema de detección automática
> del cumplimiento de equipos de protección personal — casco y chaleco —
> en obras de construcción."

**Mostrar:** slide de portada con el título.

**Tip:** no leas la slide, mirá a los profesores. Empezar firme da
confianza.

---

### Sección 2: El problema y por qué importa (~1 min)

**Qué decir:**

> "Las obras de construcción son entornos de alto riesgo. El casco y
> el chaleco son las medidas más básicas para prevenir accidentes.
> Pero la supervisión manual es costosa, intermitente y susceptible
> a sesgos humanos. Automatizar esa supervisión con visión por
> computadora puede ayudar a salvar vidas."
>
> "El problema tiene tres desafíos técnicos según el enunciado:
> oclusión (trabajadores tapados), objetos pequeños (trabajadores
> lejanos) y desbalance de clases (los casos no conformes son raros)."

**Mostrar:** slide con foto ejemplo de una obra real y los 3 desafíos
listados.

**Tip:** **no te quedes en el porqué del problema demasiado tiempo**.
Los profesores ya lo saben. 1 minuto máximo.

---

### Sección 3: Estrategia general — percepción + decisión (~1-2 min)

**Qué decir:**

> "La decisión clave del diseño es separar el sistema en dos partes:
> percepción y decisión.
>
> Por un lado, un modelo YOLO se encarga solo de DETECTAR objetos:
> cabezas con casco, cabezas sin casco, chalecos, personas. El modelo
> NO juzga, solo ve.
>
> Por otro lado, un módulo separado en Python — el verificador — toma
> esas detecciones y aplica reglas deterministas para decidir si cada
> trabajador cumple o no cumple."

> "Esta separación no es arbitraria, está pedida explícitamente por el
> enunciado y es la práctica estándar en la industria. Tres ventajas
> concretas:
>
> 1. **Auditabilidad**: sé exactamente POR QUÉ el sistema dijo "no
>    conforme" — porque puedo leer las reglas.
> 2. **Modificabilidad**: si mañana cambian las normas, modifico las
>    reglas sin reentrenar 7 horas el modelo.
> 3. **Métricas detalladas**: tengo mAP por clase, no solo accuracy
>    binario."

**Mostrar:** slide con diagrama tipo:
```
[Video/imagen] → [YOLO detecta] → [Verificador decide] → [Veredicto]
```

**Tip:** este es el momento más importante de toda la exposición. Si
el profesor entiende esta separación, todo lo demás se entiende
solo. Asegurate que quede claro.

---

### Sección 4: Datos — cómo combiné 3 fuentes (~2 min)

**Qué decir:**

> "Para entrenar el modelo combiné tres datasets públicos de Roboflow:
> Hard Hat Workers de Northeastern University, Safety Helmet, y
> Safety Vest PPE. En total, casi 26.000 imágenes."

> "Cada dataset tenía su propia taxonomía de clases — algunas con
> 'helmet', otras con 'hardhat', otras con 'safety-vest'. Lo primero
> que hice fue unificar todo a una taxonomía de 5 clases:
> head_with_helmet, head_without_helmet, person, vest, no_vest_person."

> "Esa unificación está en un YAML versionado en el repo
> (`configs/class_mapping.yaml`) — es la única fuente de verdad."

> "Después, dividí las imágenes en train/val/test con proporción
> 70/15/15, **estratificado por combinación de clases presentes en
> cada imagen**. Esto previene fuga entre splits y asegura que cada
> clase aparezca proporcionalmente en cada split."

**Mostrar:** slide con:
- Las 3 fuentes (logos o nombres)
- Tabla de la taxonomía de 5 clases
- Figura de distribución de clases por split (Figure 1 del informe)

**Si te preguntan:**
- "¿Por qué esos tres datasets?" → "Cobertura de las clases que
  necesitaba: SHWD aporta negativos explícitos de casco, Hard Hat
  aporta volumen, PPE aporta chalecos."
- "¿Por qué estratificado?" → "Para que la proporción de cada clase
  sea la misma en train, val y test. Si por azar las pocas instancias
  de `person` quedaran solo en train, no podría medir su desempeño."

---

### Sección 5: Entrenamiento — baseline (~2 min)

**Qué decir:**

> "El primer entrenamiento fue un baseline con YOLOv8s a 640 píxeles
> y configuración por defecto. 80 epochs, optimizador SGD, mosaic
> activo. Lo corrí en Kaggle con GPU T4, tardó unas 7.3 horas."

> "Los resultados del baseline ya fueron muy buenos:
> - head_with_helmet: 97.8% de mAP@0.5
> - head_without_helmet: 95.9%
> - vest: 98.7%
> - person: solo 17.5% — clase escasa, ya lo esperaba"

**Mostrar:** slide con la tabla de métricas del baseline + curvas de
entrenamiento (Figure 2 del informe).

**Si te preguntan:**
- "¿Por qué YOLOv8s y no v9 o algo más grande?" → "Por costo-beneficio
  para un parcial. YOLOv8 tiene tooling más maduro, comunidad activa,
  documentación extensa. La variante small (~11M parámetros) entra
  cómoda en la VRAM de la T4. Para producción sí evaluaría v8m o v9."
- "¿Por qué 80 epochs?" → "Suficiente para convergencia. Tenía
  early stopping con patience=20; nunca se activó porque el modelo
  siguió mejorando hasta el último epoch."

---

### Sección 6: Experimentos (~2 min)

**Qué decir:**

> "Después del baseline diseñé 4 experimentos para atacar los 3 desafíos
> del enunciado:
>
> - **A**: subir resolución a 960 → ayuda con objetos pequeños.
> - **B**: agregar Random Erasing + MixUp → simula oclusión.
> - **C**: pesos de clase → atacar el desbalance.
> - **D**: cerrar mosaic en los últimos 15 epochs → afinar con
>   imágenes naturales al final."

> "Por restricción de cuota GPU en Kaggle — 30 horas semanales, y cada
> entrenamiento son 7-8 horas — solo pude correr B y D. Tomé esas dos
> decisiones con justificación:"

> "**A lo descarté** porque a resolución 960 estimaba 12-15 horas,
> rebasa el límite de 12 horas por sesión de Kaggle."

> "**C lo descarté** porque el problema de la clase `person` no es de
> balance, es de cantidad de datos. Tengo 1.116 ejemplos de train.
> Ningún esquema de pesos puede inventar más ejemplos."

**Mostrar:** tabla comparativa de los 3 experimentos (baseline, B, D).

**Si te preguntan:**
- "¿Cuál ganó?" → "D, pero marginalmente. +0.2 pp en mAP@0.5-0.95
  consistente en todas las clases. El baseline ya estaba muy fuerte
  porque el dataset era limpio."
- "¿Y B?" → "Hallazgo interesante: las augmentaciones agresivas no
  aportaron, degradaron 0.6 pp la precisión. Esto significa que los
  datos ya contenían suficiente variabilidad natural. Es un hallazgo
  válido para reportar honestamente."

---

### Sección 7: Modelo final + test set (~2 min)

**Qué decir:**

> "Elegí el Experimento D como modelo final. Después lo evalué sobre
> el **test set** — las 3.895 imágenes que el modelo nunca había visto
> ni durante entrenamiento ni durante selección de hiperparámetros."

> "Las métricas en test fueron casi idénticas a las de validación:
> mAP@0.5 cayó solo 0.004 puntos. Esto significa que **el modelo
> generaliza correctamente, sin overfitting**."

**Mostrar:** tabla del test set con métricas por clase + matriz de
confusión (Figure 3 del informe).

**Tip:** mostrá la matriz de confusión y señalá la diagonal fuerte:
> "Acá pueden ver que las predicciones caen casi siempre en la
> diagonal correcta. La confusión entre clases es prácticamente
> cero — el modelo no se confunde casco/no-casco, ni casco/chaleco.
> Los errores son por detección perdida, no por mala clasificación."

---

### Sección 8: Análisis de oclusión — entregable 3 (~1-2 min)

**Qué decir:**

> "Para el análisis de robustez por oclusión, calculé el IoU máximo
> entre cualquier par de bboxes en cada imagen del test. Eso me da
> el nivel de oclusión por imagen. Definí 3 grupos: bajo (<15%),
> medio (15-40%) y alto (>40%)."

> "Resultados: el modelo es robusto a oclusión leve-media. La caída
> grande aparece en oclusión alta, donde el recall cae 17 puntos
> porcentuales. Es importante notar que la precision se mantiene
> — el modelo PIERDE detecciones, no inventa. Eso es el
> comportamiento más conservador para seguridad."

**Mostrar:** slide con tabla de mAP por nivel de oclusión.

**Si te preguntan:**
- "¿Solo 15 imágenes en oclusión alta?" → "Sí. Es un caveat
  estadístico que reportamos honestamente: el intervalo de confianza
  del '−7.5 pp' es amplio. La tendencia es clara pero el número
  exacto requeriría un dataset con más escenas densas."

---

### Sección 9: El verificador — corazón del proyecto (~3 min)

**Qué decir:**

> "Ahora viene el verificador. Es el módulo determinista que toma las
> detecciones del modelo y emite veredictos por trabajador."

> "Originalmente la regla del verificador era 'casco + chaleco'. Pero
> durante el desarrollo tomé una decisión importante de alcance que
> quiero defender: **redefinir la regla a solo casco**. Lo explico en
> la próxima sección."

> "La lógica final funciona así: por cada cabeza detectada el sistema
> decide:
> - Cabeza CON casco → CONFORME (verde)
> - Cabeza SIN casco → NO_CONFORME, razón: 'casco_ausente' (rojo)
>
> Y para personas detectadas sin cabeza visible:
> - INDETERMINADO (amarillo) — 'cabeza_no_visible'"

> "Quiero detenerme un momento en INDETERMINADO. **No es un error del
> sistema, es una decisión de diseño**. Cuando el modelo ve a un
> trabajador pero no puede ver su cabeza, el verificador prefiere
> admitir incertidumbre antes que inventar un veredicto. Esto es
> coherente con el principio ético: un sistema de seguridad no debe
> alucinar resultados."

> "Detalle técnico: la versión inicial del verificador usaba `person`
> como ancla. Eso falló porque la clase `person` es escasa, el modelo
> solo detecta el 17% de las personas. Tuve que reescribir la lógica
> para usar la **cabeza** como ancla, que sí es abundante."

> "Todo el módulo está cubierto por **39 tests pytest** sobre escenarios
> sintéticos."

**Mostrar:** slide con el diagrama de la lógica del verificador + la
imagen de salida de ejemplo (Figure 4 del informe).

**Si te preguntan:**
- "¿Por qué no entrenaron un clasificador directo CONFORME/NO?" →
  ver Q&A del glosario, sección 7.
- "¿Cómo decide que un chaleco va 'con' una cabeza?" → "Por
  proximidad espacial: el centro del chaleco debe estar debajo del
  centro de la cabeza, y el centro horizontal de la cabeza debe
  caer dentro del rango horizontal del chaleco. Los umbrales son
  configurables."

---

### Sección 10: Hallazgo del sesgo de datos + decisión de alcance (~2-3 min)

**Qué decir:**

> "Cuando apliqué el verificador inicial — con regla casco+chaleco — a
> imágenes reales del val set, noté que aparecía muchísimo amarillo,
> muchos INDETERMINADO. Lo investigué cuantitativamente."

> "Sobre 300 imágenes analizadas:
> - 49% el modelo solo detectaba head_with_helmet
> - 40% solo vest
> - Solo el **3.3%** tenía cabeza Y chaleco detectados simultáneamente"

> "La causa es estructural del dataset combinado: SHWD y Hard Hat
> anotaron solo cabezas, PPE anotó solo chalecos. El modelo aprendió
> la correlación 'estilo de imagen → clases visibles' y se comporta
> de manera bimodal."

> "**Frente a este diagnóstico tomé una decisión técnica de alcance**:
> redefinir el verificador para que use SOLO casco como criterio. Lo
> hice por dos razones:"

> "**Primera, contexto operativo local**. En obras peruanas pequeñas y
> medianas, el chaleco reflectivo no es uniforme universal — la prenda
> típica es polo manga larga + casco. El casco sí es obligatorio en
> todos los contextos."

> "**Segunda, limitación de datos documentada**. La regla del chaleco
> quedaba operativamente bloqueada por el sesgo de etiquetado entre
> datasets."

> "El resultado: con la regla solo casco, la tasa de INDETERMINADO
> cayó de 27% a 0.1%. El sistema produce 100% de veredictos definitivos
> sobre el mismo video."

**Mostrar:** slide con la tabla comparativa de antes/después:
- Regla casco+chaleco: 861 CONFORME, 2.600 NO_CONFORME, 1.309 INDETERMINADO
- Regla solo casco: 3.461 CONFORME, 0 NO_CONFORME, 3 INDETERMINADO

**Tip:** este es **uno de los puntos más fuertes del proyecto**:
mostrás diagnóstico → decisión informada → mejora medible. Eso es
pensamiento ingenieril. Practicalo bien.

**Si te preguntan "¿no estás reduciendo el alcance del enunciado?":**
> "Lo estoy ajustando, no reduciendo. El enunciado plantea casco y
> chaleco como contextuales — el mismo enunciado menciona que el arnés
> es 'en contextos específicos como trabajo en altura'. En el contexto
> peruano, el chaleco también es contextual. La decisión está
> justificada con dos pilares: realidad operativa local y limitación
> de datos diagnosticada. Es honesta y defendible."

---

### Sección 11: Validación experimental — modelo externo (~2 min)

**Qué decir:**

> "Para confirmar que el problema era del dataset y no de mi pipeline,
> hice un experimento de validación: usé un modelo YOLOv8 externo
> entrenado sobre un dataset bien anotado — Construction Site Safety,
> con 10 clases todas presentes en cada imagen."

> "Pasé el mismo video por ambos modelos. Mi verificador NO se
> modificó, solo se remapean las clases del modelo externo a la
> taxonomía mía antes de pasarlas al verificador."

> "Resultados sobre 773 frames del mismo video:
> - Mi modelo D: 861 CONFORME, 2.600 NO_CONFORME, 1.309 INDETERMINADO
> - Modelo externo: 2.287 CONFORME, 7.221 NO_CONFORME, **397**
>   INDETERMINADO"

> "El modelo externo produce **2.75 veces más veredictos definitivos**.
> La tasa de INDETERMINADO cae del 27% al 4%. **Esto confirma
> cuantitativamente que el cuello de botella era el dataset, no el
> pipeline**."

> "**Aclaro**: el modelo externo no es producto de este trabajo, solo
> lo uso como referencia experimental para validar mi diagnóstico.
> El modelo entregado por este parcial sigue siendo el Experimento D."

**Mostrar:** slide comparando 2 frames del mismo video procesados por
ambos modelos (Figure 5 del informe).

**Si te preguntan:**
- "¿Pero no podrían entonces usar ese modelo externo y listo?" →
  "Para producción, sí. Para este parcial, no — el deliverable tiene
  que ser un modelo que YO haya entrenado. Pero la validación es
  honesta: muestra qué resultado se obtendría con un dataset
  apropiado, lo cual es una conclusión útil del trabajo."

---

### Sección 12: Demo en video (~2 min, EL MOMENTO CLAVE)

**Qué decir antes de reproducir:**

> "Para cerrar la parte técnica, les muestro el sistema corriendo sobre
> un video real de una obra colombiana que **no fue visto durante el
> entrenamiento**. 26 segundos, una cuadrilla vertiendo concreto. Todos
> los trabajadores portan casco de seguridad amarillo."

**Reproducir el video** (`demo_video/video03_solo_casco.mp4`).

**Mientras se reproduce:**

> "Pueden ver los bboxes verdes alrededor de cada trabajador con casco
> detectado, etiquetados como CONFORME. La tasa de cumplimiento se
> actualiza por fotograma en la esquina superior izquierda."
>
> "El sistema correctamente confirma que la cuadrilla cumple al 100%
> con el casco de seguridad. Los pocos casos amarillos que aparecen
> son INDETERMINADO legítimo: trabajadores que están de espaldas o
> parcialmente ocluidos por la maquinaria, donde la cabeza no es
> visible. El sistema honestamente reporta 'no puedo decidir sin ver
> la cabeza'."

**Si el profesor te trae un video nuevo:**

> "¿Quieren que lo procese con un video de ustedes en tiempo real?"

Y mostrás cómo arrastrar el video al `procesar_video.bat`. (Tu guía
en `GUIA_VIDEO_NUEVO.md` cubre esto.)

---

### Sección 13: Implicaciones éticas (~1-2 min)

**Qué decir:**

> "La sección ética del informe cubre tres ejes obligatorios:"

> "**Primero, el balance entre falsos negativos y falsos positivos**.
> En seguridad ocupacional son asimétricos: un falso negativo deja a
> un trabajador en riesgo; un falso positivo erosiona la confianza
> en el sistema. **El umbral de confianza no debe elegirse maximizando
> mAP, sino considerando el costo asimétrico de los errores**."

> "**Segundo, la privacidad**. Un sistema que graba continuamente
> plantea consentimiento informado, uso secundario de imágenes, y
> retención de datos. El sistema técnico solo no resuelve esto,
> necesita políticas operativas."

> "**Tercero, los sesgos del dataset**. Los datos vienen
> mayoritariamente de obras en Asia, Europa y Norteamérica. No hubo
> auditoría demográfica. Para despliegue real haría falta un piloto
> en el sitio objetivo, con re-evaluación."

**Tip:** esto puede parecer relleno pero **es uno de los entregables
obligatorios**. Dedicale al menos 1 minuto y mostrá que pensaste
las implicaciones de manera concreta, no genérica.

---

### Sección 14: Conclusiones (~1-2 min)

**Qué decir:**

> "Para cerrar, las conclusiones del trabajo:"

> "1. **El modelo entrenado funciona**: mAP@0.5 = 0.771 sobre test,
>    con desempeño excelente — mayor a 0.94 — en las 3 clases
>    críticas."

> "2. **Generaliza correctamente**: val y test son casi idénticos."

> "3. **El verificador determinista es correcto**: 29 tests pasando
>    + validación cualitativa sobre val y video real."

> "4. **La principal limitación operativa es del modelo, no del
>    verificador**: el sesgo de etiquetado entre los tres datasets
>    se documentó cuantitativamente Y se validó experimentalmente
>    con un modelo externo."

> "5. **El sistema admite incertidumbre honestamente** mediante
>    INDETERMINADO, no inventa veredictos."

> "6. **El despliegue real requiere más que el código**: políticas
>    de privacidad, calibración del umbral según costo de errores,
>    auditoría de sesgos."

**Mostrar:** slide con las 6 conclusiones como bullets.

---

### Sección 15: Cierre y preguntas (~30 seg + 5 min Q&A)

**Qué decir para cerrar:**

> "Todo el código, las configuraciones, los notebooks de Kaggle, los
> tests, las tablas de resultados y el modelo final están publicados
> en mi repositorio de GitHub bajo el release v1.0. Pueden
> reproducir todo el experimento siguiendo el README. **Gracias.
> ¿Preguntas?**"

**Mostrar:** slide final con la URL del repo:
`https://github.com/PercyTechX/epp-yolo-compliance`

---

## 🎯 Preguntas que pueden hacerte (con respuestas)

Tenés un Q&A completo en `GLOSARIO_METRICAS.md` sección 7. Las más
probables:

1. **"¿Por qué YOLOv8?"** → tooling maduro, plazo definido,
   compatible con Kaggle.
2. **"¿Qué es mAP@0.5?"** → métrica estándar, IoU mínimo 0.5 para
   considerar acierto.
3. **"¿Por qué tantos INDETERMINADO?"** → sesgo de etiquetado entre
   datasets, validado experimentalmente.
4. **"¿Por qué `person` tiene 0.18 de mAP?"** → escasez de datos
   (1.116 train), no del modelo.
5. **"¿No habrían sido mejores datos sintéticos?"** → enunciado los
   descartó; pueden introducir sesgos artificiales.
6. **"¿Por qué no Cross-Validation?"** → costo de cómputo (30h
   semanales en Kaggle); split 70/15/15 con estratificación es
   suficiente para datasets grandes.

---

## ⚠️ Cosas que NO conviene decir

- **NO digas** "no sé" sin más. Decí "no lo medí en este trabajo
  pero sería un buen experimento para hacer en una versión futura".
- **NO digas** "el modelo es malo" hablando de la clase `person`.
  Decí "la clase es estructuralmente limitada por la composición de
  los datasets".
- **NO digas** "copié código de internet". Decí "usé Ultralytics,
  que es la implementación oficial de referencia, y desarrollé el
  verificador determinista desde cero con 29 tests propios".
- **NO digas** "no me alcanzó el tiempo". Decí "prioricé los
  experimentos con mejor costo-beneficio dada la cuota de cómputo
  disponible".
- **NO digas** "el OtroEnfoque es mejor que mi modelo".
  Decí "el modelo externo se usa solo como validación experimental
  para confirmar mi diagnóstico sobre el dataset".

---

## 💪 Tips finales de exposición

### Lenguaje corporal

- **Mirá a los profesores**, no a la pantalla ni al piso.
- **Pausá** después de afirmaciones importantes (deja que asienten).
- **Caminá** un poco si te sentís cómodo, no te quedes pegado al podio.

### Voz

- Habla **un poco más lento** de lo natural.
- **Subí el volumen** un escalón sobre tu voz normal.
- **Variá la entonación** — no monótono.

### Manejo de los nervios

- Antes de empezar, **respira hondo 3 veces**.
- Si te quedás en blanco, **decí "déjenme un momento, miro mis
  notas"** y consultá este guion. Es legítimo.
- Si te equivocás con un número, **corregilo sin dramatizar**: "perdón,
  era 0.771 no 0.781, sigamos".

### Tiempo

- **Llevá un reloj o el celular en vista** para no pasarte.
- Si vas adelantado, **expandí la sección de validación experimental**
  (es la más fuerte).
- Si vas atrasado, **acortá el demo a 30 segundos** y saltá a las
  conclusiones.

---

## 📋 Checklist el día anterior

- [ ] Probar el `procesar_video.bat` con cualquier video.
- [ ] Verificar que VLC abre el `video03_annotated_v2.mp4`.
- [ ] Tener el repo abierto en VSCode/browser.
- [ ] Imprimir o tener accesible este guion + el glosario.
- [ ] Slides terminadas y exportadas como PDF (backup).
- [ ] Cargar laptop al 100% + llevar cargador.
- [ ] Probar el HDMI/proyector si vas a usar el aula.
- [ ] Repasar los 6 números clave: mAP global 0.771, por clase
      0.976/0.947/0.984/0.176, generalización −0.4 pp.
- [ ] Repasar las 6 conclusiones.
- [ ] **Dormir bien**. Una buena noche vale más que repasar
      1 hora extra.

---

## 🏁 Frase para terminar bien

Si llegás al final con confianza, terminá con algo como:

> "Más allá del modelo concreto entrenado, lo que rescato de este
> trabajo es haber recorrido todo el ciclo: diagnosticar el problema,
> elegir las herramientas, entrenar, evaluar honestamente, identificar
> limitaciones y validarlas experimentalmente. Para mí ese fue el
> verdadero aprendizaje del parcial. Gracias."

Esa frase te diferencia de quien solo recita métricas.

**Mucha suerte 🚀**
