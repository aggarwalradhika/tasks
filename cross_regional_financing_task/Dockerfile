FROM apex_arena:base


# -------------- Place holder for additional dependencies --------------
RUN uv pip install --system --no-cache-dir "pandas>=2.2,<3" "numpy>=2.0,<3"
# -------------- Place holder ends --------------


# ---- DO NOT CHANGE ORDER BELOW THIS LINE ----
# Copy reference solution, test answers, and data
COPY ./grader.py /tests/
COPY ./solution.sh /tests/
COPY ./tests/answers.csv /tests/

# Make /tests readable/traversable by non-root (keep root ownership!)
RUN chmod 755 /tests && chmod 644 /tests/* \
    && chmod +x /tests/solution.sh || true

COPY --chown=model:model ./data /workdir/data

# Reference solution must be executable for `apex-arena test-solution`
RUN chmod +x /tests/solution.sh

# Run as the model user during grading
USER model
