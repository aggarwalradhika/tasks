FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && apt-get install -y --no-install-recommends \
python3-pip \
&& apt-get clean
# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (Nothing extra required here)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./tests/ /tests/
COPY ./data /workdir/data

RUN chown -R model.model /workdir/data && chmod -R 700 /workdir/data