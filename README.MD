# Alineando LLMs a la actividad cerebral

__Workshop Iberamia 2024__ 

> Este repo está basado en el [trabajo de Laurent Bonasse-Gahot](https://github.com/l-bg/llms_brain_lateralization) que acompaña al artículo [fMRI predictors based on language models of increasing complexity recover brain left lateralization](https://arxiv.org/abs/2405.17992).

Adaptación de Álvaro Cabana.

__Instrucciones de instalación__

1. Clonar este repo en una carpeta local.

2. Crear un nuevo entorno python (`conda` recomendado):

    conda create --name "brain_llms" python=3.11
    
    conda activate brain_llms
    
3. Desde la carpeta local del repo:

    pip install -r requirements.txt


__Primeros pasos (correr antes del workshop en la medida de lo posible)__

1. Extraer "activaciones" de un word embedding

    python extract_swow_activations.py

2. Alinear señal de embeddings a la actividad neural promedio

    python fit_average_subject.py --model swow

3. (Opcional) Extrar activaciones de un LLM (esto descarga ~1Gb en disco y tarda ~10 minutos)

    python extract_llm_activations.py --model Qwen2.5-0.5B 

3. (Opcional) Alinear señal de embeddings a la actividad neural promedio (esto puede demorar varias horas)

    python fit_average_subject.py --model swow
    
4. Lanzar jupyter y abrir notebook para visualizar resultados

    jupyter-notebook
