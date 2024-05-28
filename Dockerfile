# FAMLI3
#
# VERSION               golob/famli2:v3.0.1

FROM --platform=amd64 quay.io/biocontainers/biopython:1.81

RUN wget https://github.com/jgolob/FAMLI3/releases/download/v3.0.1/famli3-v3.0.1-ubuntu-latest && \
mv famli3-v3.0.1-ubuntu-latest /usr/local/bin/famli3 && \
chmod +x /usr/local/bin/famli3
