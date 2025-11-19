FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages for data science
RUN pip3 install --no-cache-dir --break-system-packages \
    scikit-learn==1.3.0 \
    scipy==1.11.1

# -------------- Placeholder ends -------------------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./data /workdir/data

# Make data readable by 'model' user
RUN chown -R model:model /workdir/data && chmod -R 700 /workdir/data