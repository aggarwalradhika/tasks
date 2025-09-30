FROM apex_arena:base

# Minimal, deterministic toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    pip3 install --no-cache-dir pandas==2.2.2 numpy==1.26.4 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy tests, solution, and data
COPY ./grader.py /tests/
COPY ./solution.sh /tests/
COPY ./data/ /workdir/data/

# Permissions: make /workdir readable by the model user
RUN chown -R model:model /workdir && chmod -R 755 /workdir && \
    chmod +x /tests/solution.sh
