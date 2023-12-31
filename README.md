# Gender-and-Age-Recognition-System-from-Speech
## A project whose objective is to recognize the age range and the gender of a person based on their speech.

Este proyecto aborda el procesamiento y análisis de señales de audio para reconocer género y edad. Se han diseñado pasos específicos de procesamiento para mejorar la calidad de las señales de audio y optimizar el rendimiento del modelo de reconocimiento.

## Índice
- [Descripción General](#descripción-general)
- [Notebook de Procesamiento de Audio](#notebook-de-procesamiento-de-audio)
- [Cómo Utilizar el Módulo de Preprocesamiento](#cómo-utilizar-el-módulo-de-preprocesamiento)
- [Ejemplos de Salida de Audio Preprocesados](#ejemplos-de-salida-de-audio-preprocesados)
- [Cómo Utilizar el Módulo de Extracción de Características](#cómo-utilizar-el-módulo-de-extracción-de-características)
- [Notebooks de Extracción de Características](#notebooks-de-extracción-de-características)

## Descripción General

El objetivo principal es procesar señales de audio para que sean aptas para modelos de reconocimiento. A través de este repositorio, se proporciona una notebook que ilustra cada paso del procesamiento y un módulo Python.

## Notebook de Procesamiento de Audio

La notebook incluida en este repositorio ilustra cada paso del procesamiento de audio, permitiendo una visualización clara de la transformación de la señal en cada etapa. 

Puedes [hacer clic aquí](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/Preprocesamiento.ipynb) para acceder a la notebook.

## Cómo Utilizar el Módulo de Preprocesamiento

El código de procesamiento ha sido adaptado para funcionar como un [módulo](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/preprocesamiento_audio.py) independiente, facilitando su implementación:

Para procesar un archivo de audio, sigue estos pasos:

1. Importa el módulo en tu script o notebook de Python:

   ```import preprocesamiento_audio```

2. Llama a la función procesar_audio con el nombre de tu archivo de audio:

    ```nombre_del_audio = "nombre_del_archivo.mp3" ```

    ```preprocesamiento_audio.procesar_audio(nombre_del_audio)```

3. Tras ejecutar la función, se generarán cinco archivos:

    - **nombre_del_archvio_con_reduccion_de_ruido.mp3:** Audio preprocesado hasta la reducción de ruido en formato MP3.
    - **nombre_del_archvio_sin_silencios_largos.mp3:** Audio preprocesado hasta la eliminación de silencios largos en formato MP3.
    - **nombre_del_archvio_normalizado.mp3:** Audio preprocesado hasta la normalización en formato MP3.
    - **nombre_del_archvio_con_preenfasis.mp3:** Audio segmentado con un filtro de preénfasis en formato NumPy.
    - **nombre_del_archvio_sin_preenfasis.mp3:** Audio segmentado sin un filtro de preénfasis en formato NumPy.

## Ejemplos de Salida de Audio Preprocesados
Para una comprensión más clara del procesamiento, hemos creado un [sitio web](https://jazminps.github.io/Gender-and-Age-Recognition-System-from-Speech/) que proporciona ejemplos auditivos del audio antes y después del procesamiento.


## Cómo Utilizar el Módulo de Extracción de Características

Los códigos de extracción de características han sido adaptados para funcionar como un [módulo](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/extraccion_de_caracteristicas.py) independiente, facilitando su implementación:

Para utilizarlo, sigue estos pasos:

1. Importa el módulo en tu script o notebook de Python:

   ```import extraccion_de_caracteristicas```

2. Llama a la función extraer_caracterisiticas con el nombre de tu archivo de audio:

    ```nombre_del_archivo = "nombre_del_archivo_que_contiene_los_frames.npy" ```

    ```extraccion_de_caracteristicas.extraer_caracterisiticas(nombre_del_archivo)```

3. Tras ejecutar la función, se generarán cuatro archivos:

    - **nombre_del_archivo_que_contiene_los_frames_lpc.npy:** Archivo de salida con coeficientes LPC.
    - **nombre_del_archivo_que_contiene_los_frames_lpcc.npy:** Archivo de salida con coeficientes LPCC.
    - **nombre_del_archivo_que_contiene_los_frames_mfcc.npy:** Archivo de salida con coeficientes MFCC.
    - **nombre_del_archivo_que_contiene_los_frames_plp.npy:** Archivo de salida con coeficientes PLP.

## Notebooks de Extracción de Características
En este repositorio, también se incluyen notebooks que ilustran la extracción de distintos coeficientes de las señales de audio, como LPC, LPCC, MFCC y PLP.

1. [Notebook de Coeficientes LPC](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/LPC.ipynb)
2. [Notebook de Coeficientes LPCC](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/LPCC.ipynb)
3. [Notebook de Coeficientes MFCC](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/MFCC.ipynb)
4. [Notebook de Coeficientes PLP](https://github.com/JazminPS/Gender-and-Age-Recognition-System-from-Speech/blob/main/PLP.ipynb)