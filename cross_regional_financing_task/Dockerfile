FROM apex_arena:base


# -------------- Place holder for additional dependencies --------------
RUN uv pip install --system --no-cache-dir "pandas>=2.1.1" "numpy>=2.2.2"
# -------------- Place holder ends --------------

# Ensure MCP server files are readable/traversable by the model user
RUN chmod -R a+rX /mcp_server

# ---- DO NOT CHANGE ORDER BELOW THIS LINE ----
# Copy reference solution, test answers, and data
COPY ./grader.py /tests/
COPY ./solution.sh /tests/
RUN chmod +x /tests/solution.sh
COPY ./tests/answers.csv /tests/

# Make /tests readable/traversable by non-root (keep root ownership!)
RUN chmod 755 /tests && chmod 644 /tests/* || true

COPY --chown=model:model ./data /workdir/data

# Run as the model user during grading
USER model
