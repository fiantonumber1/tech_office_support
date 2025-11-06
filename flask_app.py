# flask_app.py
from flask import Flask, request, jsonify
import sympy as sp
from sympy import exp, diff, simplify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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
    lam_symbols = {f"lam{i+1}": sp.symbols(f"lam{i+1}") for i in range(len(lambdas))}

    expr = sp.sympify(fungsi_str)
    for name, val in lambdas.items():
        sym = lam_symbols.get(name)
        if sym:
            expr = expr.subs(sym, val)

    # JANGAN POTONG! Biarkan full expression
    R_str = str(expr)
    h_str = "unknown"
    method = "symbolic"

    # --- COBA SIMBOLIK ---
    try:
        R = simplify(expr, seconds=2)
        h_expr = simplify(-diff(R, t) / R, seconds=2)
        h_func = sp.lambdify(t, h_expr, modules=["numpy", {"exp": safe_exp}])

        rows = []
        for t_val in t_values:
            try:
                h_val = float(h_func(t_val))
                if np.isfinite(h_val) and h_val >= 0:
                    rows.append({
                        "t": f"{t_val:.6e}",
                        "hazard_rate": f"{h_val:.6e}",
                        **lambdas
                    })
                    continue
            except:
                pass
            raise

        # FULL EXPRESSION, tanpa potong
        h_str = str(h_expr)

    except Exception as e_sym:
        method = "numerical (simplified symbolic + finite diff)"
        print(f"Symbolic hazard failed: {e_sym}. Using simplified R(t) + numerical diff.")

        R_func_raw = sp.lambdify(t, expr, modules=["numpy", {"exp": safe_exp}])
        def R_func(t_val):
            return safe_R_eval(R_func_raw, t_val)

        delta_ratio = 1e-6
        rows = []
        for t_val in t_values:
            try:
                R_t = R_func(t_val)
                if R_t >= 1.0:
                    h_val = 0.0
                elif R_t <= 0.0:
                    h_val = np.inf
                else:
                    delta = max(t_val * delta_ratio, 1e-8)
                    t_plus = t_val + delta
                    t_minus = max(t_val - delta, 1e-8)
                    R_plus = R_func(t_plus)
                    R_minus = R_func(t_minus)

                    if R_plus >= 1.0 or R_minus >= 1.0:
                        h_val = 0.0
                    elif R_plus <= 0 or R_minus <= 0:
                        h_val = np.inf
                    else:
                        dR_dt = (R_plus - R_minus) / (2 * delta)
                        h_val = max(-dR_dt / R_t, 0)

                h_val = h_val if np.isfinite(h_val) else 0.0
            except:
                h_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": f"{h_val:.6e}" if np.isfinite(h_val) else "0.000000e+00",
                **lambdas
            })

        # h(t) tidak bisa diekspresikan simbolik → beri keterangan
        h_str = "numerical: h(t) ≈ - (dR/dt) / R(t) using central difference"

    df = pd.DataFrame(rows)
    return {
        "R": R_str,           # Full, tidak dipotong
        "h": h_str,           # Full atau keterangan jelas
        "data": df.to_dict(orient="records"),
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
