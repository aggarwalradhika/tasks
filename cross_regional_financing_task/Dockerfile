FROM apex_arena:base


# -------------- Place holder for additional dependencies --------------
RUN uv pip install --system --no-cache-dir "pandas>=2.2" "numpy>=2.0" openpyxl scikit-learn
# -------------- Place holder ends --------------

# ------------ Place holder for operations --------------
# ENV HARD_MODE=1
# ------------ Place holder for operations --------------


# ---- DO NOT CHANGE ORDER BELOW THIS LINE ----
# Copy reference solution, test answers, and data
COPY ./grader.py /tests/

COPY ./tests/answers.csv /tests/
COPY --chown=model:model ./data /workdir/data

# Reference solution must be executable for `apex-arena test-solution`
RUN chmod +x /tests/solution.sh

# Run as the model user during grading
USER model
