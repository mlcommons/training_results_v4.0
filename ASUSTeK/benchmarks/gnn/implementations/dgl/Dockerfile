ARG FROM_IMAGE_NAME=nvcr.io/nvidia/dgl:24.04-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/gnn
ENV WORK_DIR=/workspace/gnn

COPY . .
# required packages not included in DGL container: 
RUN pip install -r requirements.txt

ENV WG_LOAD_USE_DIRECTIO=1

CMD /bin/bash
