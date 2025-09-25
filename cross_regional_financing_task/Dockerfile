# FROM apex_arena:base

# USER root
# COPY tests/ /tests/

# USER model
# WORKDIR /workdir

# COPY ./data /workdir/data

# USER root
# RUN uv pip install --system pandas numpy openpyxl scikit-learn

# COPY ./solution.sh /tests/
# RUN chmod +x /tests/solution.sh

# USER MODEL

# ------------ Place holder for operations --------------
# (Optional) Make the selection rules stricter during evals:
# ENV HARD_MODE=1
# ------------ Place holder for operations --------------

# ---- DO NOT CHANGE ORDER BELOW THIS LINE (keep base ordering intact) ----

# Make the reference solution available for apex CLI
COPY ./solution.sh /tests/
RUN chmod +x /tests/solution.sh

# If you copy data, prefer:
# COPY --chown=model:model ./data /workdir/data

# Run as model during grading/solution
USER model
