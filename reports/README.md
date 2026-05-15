# Reportes

Carpeta destino de tablas y figuras generadas por los scripts de análisis
y por los notebooks de entrenamiento.

## Estructura

```
reports/
├── figures/    # PNG/PDF para el informe (gitignored)
└── tables/     # CSV con métricas y distribuciones (gitignored)
```

Los subdirectorios están en `.gitignore`. Solo se versiona este README
para preservar la estructura. Si una figura es crítica para el informe
final, copiarla al documento (LaTeX/Docs/Word) en lugar de versionarla.

## Qué se genera dónde

| Origen | Salida |
|---|---|
| `src/data/class_distribution_report.py` | `tables/class_distribution.csv`, `figures/class_distribution.png` |
| `src/data/validate_visually.py` | `figures/validation_<dataset>/...` |
| Notebooks de entrenamiento (Kaggle) | Se descargan del kernel; tablas comparativas a `tables/` |
| Evaluación final (fase 5) | `tables/test_metrics.csv`, matriz de confusión y desglose por oclusión |
