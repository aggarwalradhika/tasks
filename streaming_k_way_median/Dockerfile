FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3=3.10.* python3-pip && \
    pip3 install --no-cache-dir pandas==2.2.2 numpy==1.26.4 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (Nothing needed here for runtime operations)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py   /tests/
COPY --chown=model:model ./data /workdir/data

RUN chown -R model:model /workdir && chmod -R 755 /workdir && \
    chmod +x /tests/solution.sh
