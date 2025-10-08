FROM apex_arena:base

# -------------- Placeholder for additional dependencies --------------
# (Intentionally empty — do NOT install test-only deps like pydantic)
# -------------- Placeholder ends -------------------------------------

# -------------- Placeholder for operations ---------------------------
# (No runtime ops needed)
# -------------- Placeholder for operations ---------------------------

# -------------------- DO NOT CHANGE BELOW ----------------------------
COPY ./solution.sh /tests/
COPY ./grader.py  /tests/
COPY ./data /workdir/data

RUN chown -R model:model /workdir/data && chmod -R 755 /workdir/data
