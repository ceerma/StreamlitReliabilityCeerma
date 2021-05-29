
# mata procesos anteriores (si estaba desde otro lado el server)
pgrep stream | xargs -r kill

# inicia virtualenv
source env/bin/activate

# lanza el proceso con nohup para evitar que se muera al cerrar terminal
# y manda a background
nohup streamlit run --server.enableXsrfProtection=false --server.enableCORS false --server.baseUrlPath=parametricmodel parametricmodel.py
