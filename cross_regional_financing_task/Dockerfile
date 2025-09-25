# # FROM apex_arena:base

# # USER root
# # COPY tests/ /tests/

# # USER model
# # WORKDIR /workdir

# # COPY ./data /workdir/data

# # USER root
# # RUN uv pip install --system pandas numpy openpyxl scikit-learn

# # COPY ./solution.sh /tests/
# # RUN chmod +x /tests/solution.sh

# # USER MODEL

# # ------------ Place holder for operations --------------
# # (Optional) Make the selection rules stricter during evals:
# # ENV HARD_MODE=1
# # ------------ Place holder for operations --------------

# # ---- DO NOT CHANGE ORDER BELOW THIS LINE (keep base ordering intact) ----

# # Make the reference solution available for apex CLI
# COPY ./solution.sh /tests/
# RUN chmod +x /tests/solution.sh

# # If you copy data, prefer:
# # COPY --chown=model:model ./data /workdir/data

# # Run as model during grading/solution
# USER model
# syntax=docker/dockerfile:1


FROM apex_arena:base

# -------------- Place holder for additional dependencies --------------
# (optional) Example:
# RUN apt-get update && apt-get install -y --no-install-recommends dos2unix \
#     && apt-get clean
# -------------- Place holder ends --------------

# ------------ Place holder for operations --------------
# (optional) You can add env flags used by your solution/grader. Example:
# ENV HARD_MODE=1
# ------------ Place holder for operations --------------

# ---- DO NOT CHANGE ORDER BELOW THIS LINE ----
# Copy grader, reference solution, test answers, and data
COPY ./grader.py /tests/
COPY ./solution.sh /tests/
COPY ./tests/answers.csv /tests/
COPY --chown=model:model ./data /workdir/data

# Reference solution must be executable for `apex-arena test-solution`
RUN chmod +x /tests/solution.sh

# Run as the model user during grading
USER model
