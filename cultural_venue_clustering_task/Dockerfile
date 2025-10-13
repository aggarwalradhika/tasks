FROM apex_arena:base

USER root
COPY tests/ /tests/

USER model
WORKDIR /workdir

COPY ./data /workdir/data

USER root
RUN uv pip install --system pandas numpy openpyxl scikit-learn