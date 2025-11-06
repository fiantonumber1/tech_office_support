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
    lam_symbols = {f"lam{i+1}": sp.symbols(f"lam{i+1}") for i in range(len(lambdas))}

    # Parse fungsi dari string
    expr = sp.sympify(fungsi_str, locals=lam_symbols)

    # Substitusi nilai lambda
    for name, val in lambdas.items():
        sym = lam_symbols.get(name)
        if sym:
            expr = expr.subs(sym, val)

    # Gunakan ekspresi ASLI (tanpa simplify) untuk R(t)
    R_str = str(expr)  # Full, seperti input
    R_raw = expr       # Simpan versi mentah untuk diff
    h_str = "unknown"
    method = "symbolic"

    # --- COBA SIMBOLIK TANPA SIMPLIFY ---
    try:
        # JANGAN simplify R(t), tapi tetap hitung turunan
        h_expr = -diff(R_raw, t) / R_raw

        # Coba lambdify langsung dari h_expr mentah
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
            except Exception as e:
                print(f"Error lambdify at t={t_val}: {e}")
            raise  # Jika gagal di satu t, fallback

        # Gunakan ekspresi h(t) yang belum di-simplify (tapi tetap rapi)
        h_str = sp.latex(h_expr) if len(str(h_expr)) > 500 else str(h_expr)
        if len(h_str) > 1000:
            h_str = str(h_expr)[:1000] + "..."  # Hanya untuk JSON, bukan matematis

        method = "symbolic (no simplify)"

    except Exception as e_sym:
        # --- FALLBACK: Numerical ---
        method = "numerical (raw R(t) + finite diff)"
        print(f"Symbolic hazard failed: {e_sym}. Using raw R(t) + numerical diff.")

        R_func_raw = sp.lambdify(t, R_raw, modules=["numpy", {"exp": safe_exp}])
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

                    if not (0 < R_plus < 1 and 0 < R_minus < 1):
                        h_val = np.inf if R_t < 1e-8 else 0.0
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

        h_str = "numerical: h(t) = - (dR/dt) / R(t) [central difference, no simplify]"

    df = pd.DataFrame(rows)
    return {
        "R": R_str,           # EKSPRESI ASLI (tidak disimplify)
        "h": h_str,           # h(t) simbolik atau keterangan
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
