FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (nothing needed)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/    

COPY --chown=model:model ./data /workdir/data
RUN chmod -R 700 /workdir/data