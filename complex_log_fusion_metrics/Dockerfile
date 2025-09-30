FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip bzip2 gzip && \
    pip3 install --no-cache-dir pandas==2.2.2 numpy==1.26.4 python-dateutil==2.9.0.post0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (Nothing needed here for runtime operations)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./data /workdir/data

# Make data accessible to 'model' user (tests/ is already root-only)
RUN chown -R model.model /workdir/data && chmod -R 755 /workdir/data
# --------------------------------------------------------------------
