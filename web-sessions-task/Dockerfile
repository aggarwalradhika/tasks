FROM apex_arena:base

# --- Root: grader & tests (kept private) ---
USER root
WORKDIR /

# Grader + expected file(s) live under /tests and are root-only
COPY grader.py /tests/grader.py
COPY tests/ /tests/
RUN chmod -R 700 /tests

# --- Model: workspace (visible to agent) ---
USER model
WORKDIR /workdir

# Create writable dirs as model so no chown is required
RUN mkdir -p /workdir/data /workdir/out

# Inputs visible to the agent
COPY --chown=model:model task.yaml /workdir/task.yaml
COPY --chown=model:model data/access.log /workdir/data/access.log
COPY --chown=model:model data/policy.json /workdir/data/policy.json

# Stay as rootcle for runtime
USER root
