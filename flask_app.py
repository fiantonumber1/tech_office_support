from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
import pandas as pd

app = Flask(__name__)

# ------------------------------------------------------------------
# 1. Fungsi bantu – tetap aman dari overflow
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2. FUNGSI UTAMA – DENGAN PENCEGAHAN NEGATIF YANG SANGAT KETAT
# ------------------------------------------------------------------
def calculate_reliability(fungsi_str, lambdas, t_values):
    # ------------------------------------------------------------------
    # Validasi dasar
    # ------------------------------------------------------------------
    if any(t < 0 for t in t_values):
        raise ValueError("Semua nilai t harus ≥ 0")

    t = sp.symbols('t')
    lam_symbols = {name: sp.symbols(name) for name in lambdas.keys()}
    locals_dict = {**lam_symbols, 'exp': sp.exp}

    # ------------------------------------------------------------------
    # Parse fungsi R(t)
    # ------------------------------------------------------------------
    try:
        expr = sp.sympify(fungsi_str, locals=locals_dict)
    except Exception as e:
        raise ValueError(f"Parse error: {e}")

    R_str = fungsi_str
    h_str = "unknown"
    method = "symbolic"

    rows = []

    # ------------------------------------------------------------------
    # 2A. METODE SIMBOLIK (evalf) – paling akurat
    # ------------------------------------------------------------------
    try:
        expr_simp = sp.simplify(expr)
        expr_exp = sp.expand(expr_simp)
        h_expr = -sp.diff(expr_exp, t) / expr_exp

        for t_val in t_values:
            try:
                h_val = float(h_expr.subs(t, t_val).evalf())
                if not np.isfinite(h_val):
                    h_val = 0.0
                # PAKSA NON-NEGATIF
                h_val = max(h_val, 0.0)
            except:
                h_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": f"{h_val:.6e}",
                **lambdas
            })

        h_str = str(h_expr)
        method = "symbolic (evalf)"

    # ------------------------------------------------------------------
    # 2B. METODE NUMERIK – fallback bila simbolik gagal
    # ------------------------------------------------------------------
    except Exception as e_sym:
        method = "numerical"
        print(f"Symbolic failed: {e_sym}. Switching to numerical.")

        # Substitusi nilai lambda dulu
        expr_num = expr.subs(lambdas)
        R_func_raw = sp.lambdify(t, expr_num,
                                 modules=["numpy", {"exp": safe_exp}])

        def R_func(t_val):
            return safe_R_eval(R_func_raw, t_val)

        # Validasi R(t) di semua titik (harus 0 ≤ R ≤ 1)
        for t_val in t_values:
            r = R_func(t_val)
            if not (0 <= r <= 1.0):
                raise ValueError(f"R({t_val}) = {r:.6e} tidak valid (harus 0 ≤ R(t) ≤ 1)")

        delta_ratio = 1e-6
        for t_val in t_values:
            try:
                R_t = R_func(t_val)

                if R_t >= 1.0:                     # t = 0 atau R(t) = 1
                    h_val = 0.0
                elif R_t <= 0.0:                   # sudah rusak total
                    h_val = np.inf
                else:
                    delta = max(t_val * delta_ratio, 1e-10)
                    t_plus  = t_val + delta
                    t_minus = max(t_val - delta, 1e-12)

                    R_plus  = R_func(t_plus)
                    R_minus = R_func(t_minus)

                    # Jika ada nilai di luar [0,1] → anggap tidak stabil
                    if not (0 < R_plus < 1 and 0 < R_minus < 1):
                        h_val = np.inf if R_t < 1e-12 else 0.0
                    else:
                        dR_dt = (R_plus - R_minus) / (2 * delta)
                        raw_h = -dR_dt / R_t

                        # ---- PENCEGAHAN NEGATIF YANG SUPER KETAT ----
                        h_val = float(raw_h)
                        if not np.isfinite(h_val):
                            h_val = 0.0
                        else:
                            h_val = max(h_val, 0.0)   # <--- TIDAK PERNAH NEGATIF

                # Final safeguard
                h_val = 0.0 if not np.isfinite(h_val) else h_val
                h_val = max(h_val, 0.0)

            except Exception:
                h_val = 0.0

            h_str_val = f"{h_val:.6e}" if np.isfinite(h_val) else "0.000000e+00"
            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": h_str_val,
                **lambdas
            })

        h_str = "numerical: h(t) ≈ -dR/dt / R(t) [forced ≥ 0]"

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    return {
        "R": R_str,
        "h": h_str,
        "data": rows,                         # langsung list of dict (lebih ringan)
        "method": method,
        "warning": "Hazard rate dipaksa ≥ 0 untuk semua kasus."
    }

# ------------------------------------------------------------------
# 3. Endpoint Flask
# ------------------------------------------------------------------
@app.route('/calculate_hazard', methods=['POST'])
def calc():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000])

    # Pastikan t_values berupa list float
    try:
        t_values = [float(t) for t in t_values]
    except:
        return jsonify({"error": "t_values harus berupa list angka"}), 400

    try:
        result = calculate_reliability(fungsi, lambdas, t_values)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# 4. Run server
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5632, debug=False)   # debug=False lebih aman di prod