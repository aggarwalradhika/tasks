FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends bzip2 gzip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m ensurepip --upgrade && \
    python3 -m pip install --no-cache-dir --upgrade "pip==24.2" && \
    python3 -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "scipy==1.11.4" 
# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (Nothing needed here for runtime operations)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./tests/ground_truth.json /tests/
COPY ./data /workdir/data

# Make data accessible to 'model' user (tests/ is already root-only)
RUN chown -R model:model /workdir/data && chmod -R 755 /workdir/data
# --------------------------------------------------------------------