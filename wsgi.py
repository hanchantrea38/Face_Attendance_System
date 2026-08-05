"""WSGI entry point.

Used by production servers: gunicorn (`gunicorn wsgi:app`),
PythonAnywhere, mod_wsgi, etc.
"""
from app import app
