#!/bin/bash
set -e

echo "⏳ Ejecutando ingesta DJIA..."
python src/ingest/ingest_djia.py

echo "🧹 Limpiando documentos basura..."
python scripts/health_check.py --clean

echo "✅ Hecho. Base de datos limpia y actualizada."
