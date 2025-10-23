FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (Nothing extra required here)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./churn_logic.py /workdir/
COPY ./churn_logic.py /tests/
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./data /workdir/data

# Make data readable by 'model' user (tests/ is already root-only)
RUN chown -R model.model /workdir/data && chmod -R 755 /workdir/data