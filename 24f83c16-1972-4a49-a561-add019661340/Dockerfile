FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas==2.2.2 numpy==1.26.4

# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (nothing needed)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/    
COPY ./tests/ /tests/

COPY --chown=model:model ./data /workdir/data
RUN chmod -R 755 /workdir/data
