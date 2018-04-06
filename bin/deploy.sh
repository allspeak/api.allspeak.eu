#!/usr/bin/env bash

git reset --hard
git pull
chmod +x ./bin/*
source env/bin/activate
cd web
pip install --no-cache-dir -r requirements.txt
export FLASK_APP=project/__init__.py
flask db upgrade
deactivate