FROM apex_arena:base

RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

    
COPY ./grader.py /tests/
COPY ./solution.sh /tests/
COPY ./data/ /workdir/data/

RUN chown -R model:model /workdir && chmod -R 755 /workdir && \
    chmod +x /tests/solution.sh
