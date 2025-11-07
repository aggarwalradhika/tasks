FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------- Placeholder ends -------------------------------------
# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./data /workdir/data

# Make data readable by 'model' user (tests/ is already root-only)
RUN chown -R model:model /workdir/data && chmod -R 755 /workdir/data