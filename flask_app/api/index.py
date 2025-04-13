from vercel_wsgi import run_wsgi
from app import app

handler = run_wsgi(app)
