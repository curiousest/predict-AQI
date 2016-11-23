FROM jupyter/datascience-notebook

USER root

RUN apt-get update && apt-get install -y --no-install-recommends python-psycopg2
RUN conda install --quiet --yes psycopg2

USER $NB_USER

RUN pip3 install psycopg2
