FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
# (None needed; use Python stdlib + preinstalled)
# -------------- Placeholder ends --------------

# -------------- Placeholder for operations --------------
# (None)
# -------------- Placeholder for operations --------------

# ---------- DO NOT CHANGE BELOW THIS LINE ----------
COPY ./solution.sh /tests/
COPY ./grader.py /tests/
COPY ./data /workdir/data

RUN chown -R model.model /workdir/data && \
    chmod -R 755 /workdir/data
