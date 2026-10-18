# Veyra pública — paquete para Render

Este paquete adapta Veyra a un servicio Python pequeño. Incluye el clasificador y el generador experimental ya entrenados; el generador usa inferencia NumPy, no PyTorch, para ajustarse a la memoria del plan gratuito. No incluye la base de datos, documentos, mensajes ni otros datos del Veyra local.

## Lo que ofrece el paquete

- Web Python servida por HTTPS con sesiones separadas por pestaña del navegador.
- Chat del clasificador propio; un límite claro cuando la intención no se reconoce.
- Continuación experimental de texto del GRU propio.
- Búsqueda DuckDuckGo Lite y lectura de enlaces HTTP/HTTPS, cuando el alojamiento permite la conexión externa.
- Documentos temporales por sesión, hasta 3 archivos de 3 MB cada uno. Se eliminan al borrar la sesión o al reiniciarse la instancia.
- Voz por turnos mediante `SpeechRecognition` y `speechSynthesis` del navegador. El botón Iniciar voz solicita permiso del micrófono; el reconocimiento del navegador puede enviar audio al servicio de voz del proveedor y necesita Internet. El texto reconocido se procesa en Veyra. No se almacena audio. Detener y Silenciar cancelan la lectura; no es una llamada de tiempo real con interrupciones simultáneas.
- Sin cuentas de visitantes, entrenamiento automático, base de datos persistente, recopilación de historial, claves API ni documentos compartidos entre sesiones.

El despliegue gratuito de Render tiene 512 MB de RAM, almacenamiento efímero, duerme tras 15 minutos sin visitas y ofrece 750 horas de instancia por espacio de trabajo al mes. Una visita tras el reposo puede esperar alrededor de un minuto. No se garantiza continuidad ni indexación. Para evitar que el plan supere lo gratuito, no añadas un método de pago; Render indica que suspende los servicios si se agotan los límites de uso gratuitos sin método de pago. La configuración no solicita un plan pago.

## Archivos del paquete

- `app.py`, `veyra.py`, `public_engine.py`: servidor público, interfaz y ejecución de modelos.
- `data/modelo_veyra_propio.json`: pesos del clasificador propio.
- `data/modelo_generativo_publico.json`: pesos del GRU convertidos a NumPy.
- `data/modelo_generativo_veyra.json`: resumen verificable del entrenamiento.
- `render.yaml`: instancia gratuita, instalación de dependencias, arranque y ruta de salud.

El servicio utiliza un token de sesión aleatorio guardado en `sessionStorage`; cada pestaña mantiene documentos y conversación en memoria aislada. El servidor no escribe historiales ni documentos en SQLite. La carpeta de cargas es temporal del servicio y se limpia cuando expira una sesión o al reiniciarse. Usa datos de prueba al evaluar el sitio público y no subas información confidencial.

## Publicar

1. Crea un repositorio público de GitHub nuevo y vacío, y sube **sólo el contenido de esta carpeta**. No subas `outputs/veyra/data/veyra.sqlite3`, la carpeta local `data/documents`, archivos de `work/` ni documentos personales.
2. Inicia sesión en Render, crea un Web Service desde ese repositorio, selecciona la plantilla `render.yaml` y el plan `Free`. Comprueba el costo indicado antes de crear el servicio y deja el método de pago vacío para que el proyecto se suspenda si excede las cuotas gratuitas.
3. Render construirá la aplicación y entregará una URL `https://….onrender.com`. Para el sitemap, configura la variable `VEYRA_PUBLIC_URL` con esa URL base completa y vuelve a desplegar.
4. Verifica `/health`, la web, `/robots.txt`, `/sitemap.xml`, voz, una búsqueda y un documento temporal. Cada visitante empieza con una sesión independiente; los datos se pierden al cerrar sesión, expirar o reiniciar el servicio.

Para reconstruir el paquete después de volver a entrenar el modelo local, ejecuta en la raíz de `outputs/veyra`:

```powershell
python deploy/render/export_generativo.py
```

## Google Search

La web pública y la aparición en Google son pasos distintos. La página es rastreable y sirve un sitemap una vez configurada `VEYRA_PUBLIC_URL`. Después de publicar, verifica el dominio en Google Search Console con sus instrucciones y solicita el rastreo o envía `/sitemap.xml`. Google no garantiza indexación ni la completa inmediatamente; el servicio gratuito puede dormir y Render responde temporalmente con `robots.txt` de bloqueo mientras está suspendido.

La consulta web le pasa la búsqueda a DuckDuckGo Lite; la voz puede pasar audio al proveedor de reconocimiento del navegador. El alojamiento no entrena ni almacena un modelo con esos datos. Buscar información o subir un documento no modifica los pesos de Veyra.
