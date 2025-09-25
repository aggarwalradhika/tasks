python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("/workdir/data/financing.csv")
result = df.groupby("region")["amount"].sum().reset_index()
result.to_csv("/workdir/output.csv", index=False)
EOF
