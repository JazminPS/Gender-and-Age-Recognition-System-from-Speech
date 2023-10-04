# Gender-and-Age-Recognition-System-from-Speech
## A project whose objective is to recognize the age range and the gender of a person based on their speech.

Este proyecto aborda el procesamiento y análisis de señales de audio para reconocer género y edad. Se han diseñado pasos específicos de procesamiento para mejorar la calidad de las señales de audio y optimizar el rendimiento del modelo de reconocimiento.

## Índice
- [Descripción General](#descripción-general)
- [Notebook de Procesamiento de Audio](#notebook-de-procesamiento-de-audio)
- [Cómo Utilizar el Módulo](#cómo-utilizar-el-módulo-de-preprocesamiento)
- [Ejemplos de Salida de Audio](#ejemplos-de-salida-de-audio-preprocesados)

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

## Ejemplos de Salida de Audio Preprocesadas
Para una comprensión más clara del procesamiento, hemos creado un [sitio web](https://jazminps.github.io/Gender-and-Age-Recognition-System-from-Speech/) que proporciona ejemplos auditivos del audio antes y después del procesamiento.