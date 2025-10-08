FROM apex_arena:base

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    pip3 install --no-cache-dir typing-extensions==4.12.2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./grader.py /tests/
COPY ./solution.sh /tests/
COPY ./data/ /workdir/data/

RUN chown -R model:model /workdir && chmod -R 755 /workdir && \
    chmod +x /tests/solution.sh
