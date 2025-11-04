#!/bin/bash

# Configuration de l'environnement
export PYTHONHASHSEED=0
export FLASK_ENV=production

echo "=========================================="
echo "🚀 Design4Green 2025 - Résumer mieux avec moins"
echo "=========================================="

# Vérification de l'environnement virtuel
if [ ! -d ".venv" ]; then
    echo "🔧 Création de l'environnement virtuel..."
    python -m venv .venv
fi

echo "🔧 Activation de l'environnement..."
source .venv/bin/activate

echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo "🌐 Lancement de l'application..."
echo "📍 API: http://127.0.0.1:5000"
echo "📍 Interface: http://127.0.0.1:5000"
echo "=========================================="

python app.py
