FROM apex_arena:base

USER root
COPY tests/ /tests/

USER model
WORKDIR /workdir

COPY ./data /workdir/data

USER root
RUN uv pip install --system pandas numpy openpyxl scikit-learn

COPY ./solution.sh /tests/
RUN chmod +x /tests/solution.sh

USER MODEL