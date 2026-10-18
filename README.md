# Veyra

Asistente experimental en español. El código y los modelos pequeños propios están en [`veyra-render-free/`](veyra-render-free/). El repositorio no contiene conversaciones, documentos personales ni la base de datos del desarrollo local.

## Publicar en Render

Este repo incluye un Blueprint con plan Free. En Render, crea un Blueprint desde este repositorio y revisa que el costo seleccionado sea $0 antes de aplicar. La compilación instala las dependencias y la instancia arranca el servidor Python.

El servicio no guarda mensajes o documentos de forma persistente. Cada pestaña tiene una sesión temporal aislada; los archivos se limitan a 3 por sesión y 3 MB cada uno. Render Free duerme cuando no recibe tráfico y puede tardar en despertar. No agregues un método de pago si quieres que el servicio se suspenda al superar los límites gratuitos.

La voz opera por turnos. El reconocimiento de voz del navegador puede transmitir audio a su proveedor; no se almacena audio en el servicio. La búsqueda usa DuckDuckGo Lite sólo cuando se solicita. Los documentos son temporales y no se usan para entrenar. El generador propio es experimental y puede producir texto incoherente.

La indexación de Google no se garantiza. Para habilitar el sitemap tras el despliegue, establece `VEYRA_PUBLIC_URL` con la URL `https://…onrender.com` y despliega otra vez.
