FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (Nothing needed here for runtime operations)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py   /tests/

COPY --chown=model:model ./data /workdir/data

RUN chown -R model:model /workdir && chmod -R 700 /workdir && \
    chmod +x /tests/solution.sh && \
    chmod 0700 /tests