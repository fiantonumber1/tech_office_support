# flask_app.py
from flask import Flask, request, jsonify
import sympy as sp
from sympy import exp, diff, simplify
import numpy as np
import pandas as pd

app = Flask(__name__)

def safe_exp(x):
    x = np.asarray(x)
    return np.exp(-np.clip(x, -700, 700))

def safe_R_eval(R_func, t_val):
    try:
        result = float(R_func(t_val))
        if not np.isfinite(result):
            return 1.0
        return max(min(result, 1.0), 0.0)
    except:
        return 1.0

def calculate_reliability(fungsi_str, lambdas, t_values):
    t = sp.symbols('t')
    
    # Simbol dari lambdas (dinamis)
    lam_symbols = {name: sp.symbols(name) for name in lambdas.keys()}
    locals_dict = {**lam_symbols, 'e': sp.E}  # TAMBAHKAN e

    try:
        expr = sp.sympify(fungsi_str, locals=locals_dict)
        expr_num = expr.subs(lambdas)  # untuk numerical
    except Exception as e:
        raise ValueError(f"Parse error: {e}")

    R_str = fungsi_str
    h_str = "unknown"
    method = "symbolic"

    try:
        h_expr = -sp.diff(expr, t) / expr
        h_func = sp.lambdify(t, h_expr, modules=["numpy", {"exp": safe_exp}])

        rows = []
        for t_val in t_values:
            h_val = float(h_func(t_val))
            if np.isfinite(h_val) and h_val >= 0:
                rows.append({
                    "t": f"{t_val:.6e}",
                    "hazard_rate": f"{h_val:.6e}",
                    **lambdas
                })
            else:
                raise ValueError()
        
        h_str = str(h_expr)
        method = "symbolic"

    except Exception as e_sym:
        method = "numerical"
        print(f"Symbolic failed: {e_sym}")

        R_func_raw = sp.lambdify(t, expr_num, modules=["numpy", {"exp": safe_exp}])
        def R_func(t_val):
            return safe262_R_eval(R_func_raw, t_val)

        rows = []
        delta_ratio = 1e-6
        for t_val in t_values:
            # ... numerical diff seperti sebelumnya
            pass

        h_str = "numerical approximation"

    return {
        "R": R_str,
        "h": h_str,
        "data": pd.DataFrame(rows).to_dict(orient="records"),
        "method": method
    }

@app.route('/calculate_hazard', methods=['POST'])
def calc():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000])
    title = data.get('title', 'Hazard Rate')

    try:
        result = calculate_reliability(fungsi, lambdas, t_values)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 FLASK SERVER HIDUP! Akses di http://<IP_SERVER>:5632")
    app.run(host='0.0.0.0', port=5632, debug=True)
