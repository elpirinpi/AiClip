# AiClip

AiClip convierte vídeos y enlaces de YouTube en clips verticales con subtítulos palabra por palabra y miniaturas. La aplicación se ejecuta en Windows y abre su interfaz local en el navegador.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Claves propias para [Groq](https://console.groq.com/keys) y [Google AI Studio](https://aistudio.google.com/apikey).
- Conexión a Internet para analizar vídeos con los proveedores.

FFmpeg se incluye en las compilaciones de Windows. La aceleración NVENC se detecta automáticamente; si no está disponible, se usa x264 en CPU.

## Ejecutar desde el código fuente

1. Instala Python 3.11 y FFmpeg.
2. Crea y activa un entorno virtual, e instala las dependencias:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Copia `config.example.json` como `config.local.json`, añade tus claves y comprueba la instalación:

   ```powershell
   Copy-Item config.example.json config.local.json
   python -m scripts.check_setup
   ```

4. Inicia AiClip:

   ```powershell
   python -m scripts.start_app
   ```

La primera ejecución de la aplicación empaquetada crea una configuración sin claves en `%LOCALAPPDATA%\AiClip`. Introduce tus claves en **Ajustes**. Las claves y los datos de la aplicación se guardan en ese PC y no se incluyen en el repositorio ni en las compilaciones.

Al iniciar, AiClip consulta los releases públicos de GitHub para avisar si hay una nueva versión. Esta consulta no envía vídeos ni claves de API. El enlace descarga el instalador MSI; debes abrirlo e iniciar la instalación manualmente.

YouTube puede limitar temporalmente las solicitudes y mostrar un error HTTP 429 o pedir una verificación. En ese caso, inténtalo más tarde o selecciona el vídeo como archivo local; AiClip no utiliza las cookies del navegador.

## Privacidad y derechos

El vídeo permanece en el PC. Solo se envían a Groq fragmentos de audio mono de 16 kHz para transcribirlos y a Google Gemini el texto de la transcripción para proponer clips y títulos. Se requiere confirmación en la aplicación antes de iniciar un trabajo. Cada usuario es responsable de tener los derechos necesarios para procesar el contenido.

## Estado

Las fases de análisis, renderizado, recorte vertical, miniaturas e interfaz local están implementadas. La ingestión de streams en directo o grabados todavía no está implementada.

## Compilar para Windows

Con Python 3.11, FFmpeg estático (incluidos `ffmpeg.exe` y `ffprobe.exe`) y WiX Toolset 4.x instalados:

```powershell
python -m pip install -r requirements.txt
python -m scripts.build_windows
```

Los artefactos se generan en `dist\AiClip.exe`, `dist\FFmpeg-LICENSE.txt` y `dist\AiClip-Setup.msi`. Las compilaciones usan FFmpeg GPL de [BtbN](https://github.com/BtbN/FFmpeg-Builds); consulta sus avisos de licencia y las condiciones de redistribución de FFmpeg antes de redistribuir un paquete.

## Licencia

AiClip se distribuye bajo GPL-3.0. Consulta [LICENSE](LICENSE).
