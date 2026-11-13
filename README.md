# AiClip

AiClip convierte vídeos y enlaces de YouTube en clips verticales con subtítulos palabra por palabra y miniaturas. La aplicación se ejecuta en Windows y abre su interfaz local en el navegador.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Una clave propia de [Groq](https://console.groq.com/keys), utilizada para transcripción, análisis de momentos y textos de miniaturas.
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

Al terminar el análisis, cada clip muestra una vista previa del tramo original. Los enlaces completos, cortos (`youtu.be`), de Shorts y los enlaces de YouTube sin protocolo se aceptan. YouTube puede denegar temporalmente el acceso con HTTP 429 o pedir verificación; AiClip no elude esas restricciones ni usa cookies. Si ocurre, inténtalo más tarde o selecciona un archivo local al que tengas derecho de acceso.

Para reutilizar un vídeo de YouTube descargado, abre su proyecto y pulsa **Guardar vídeo original**. En navegadores compatibles se abre el selector para elegir la carpeta y el nombre; si no, el navegador lo guarda en su carpeta de descargas. La copia elegida no se elimina al limpiar el historial.

El análisis genera candidatos a clips de forma local con reglas ligeras sobre las marcas de tiempo y el texto; Groq GPT-OSS 20B selecciona y titula esos momentos, sin recibir la transcripción completa para esta etapa. La cantidad objetivo depende de la duración, y los candidatos se distribuyen a lo largo del vídeo sin solaparse. Si Groq devuelve HTTP 413 o el límite de tokens por minuto, AiClip divide el grupo de candidatos y vuelve a intentarlo.

Cada exportación crea archivos de clip y miniatura con nombres únicos; volver a exportar conserva los archivos anteriores. La vista previa y el ZIP usan la exportación más reciente de cada clip.

Los trabajos que terminan con error se pueden volver a poner en cola desde su pantalla. **Limpiar historial** elimina los trabajos y sus archivos descargados o temporales, pero conserva los clips y miniaturas exportados. No se puede limpiar mientras haya trabajos activos.

Al iniciar, AiClip consulta los releases públicos de GitHub para avisar si hay una nueva versión. Esta consulta no envía vídeos ni claves de API. El enlace descarga el instalador MSI; debes abrirlo e iniciar la instalación manualmente.

YouTube puede limitar temporalmente las solicitudes y mostrar un error HTTP 429 o pedir una verificación. En ese caso, inténtalo más tarde o selecciona el vídeo como archivo local; AiClip no utiliza las cookies del navegador.

## Privacidad y derechos

El vídeo permanece en el PC. Se envían a Groq fragmentos de audio mono de 16 kHz para transcribirlos. La búsqueda local de candidatos usa la transcripción en este PC; para elegir y titular clips, se envían a Groq solo los textos y tiempos de los candidatos seleccionados, no la transcripción completa. El texto pertinente también puede enviarse para generar títulos de miniatura. Se requiere confirmación en la aplicación antes de iniciar un trabajo. Cada usuario es responsable de tener los derechos necesarios para procesar el contenido.

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
