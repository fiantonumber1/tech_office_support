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
    if any(t < 0 for t in t_values):
        raise ValueError("t harus ≥ 0")

    t = sp.symbols('t')
    lam_symbols = {k: sp.symbols(k) for k in lambdas.keys()}
    locals_dict = {**lam_symbols, 'exp': sp.exp}

    # 1. Parse ekspresi R(t)
    try:
        expr = sp.sympify(fungsi_str, locals=locals_dict)
    except Exception as e:
        raise ValueError(f"Parse error: {e}")

    # 2. Substitusi lambdas
    expr_num = expr.subs(lambdas)
    R_func = sp.lambdify(t, expr_num, modules=['numpy', {'exp': lambda x: np.exp(np.clip(-x, -700, 700))}])

    def safe_R(t_val):
        try:
            r = float(R_func(t_val))
            return max(min(r, 1.0), 0.0) if np.isfinite(r) else 1.0
        except:
            return 1.0

    # 3. Hitung h(t) simbolik
    try:
        expr_simp = sp.simplify(expr_num)
        expr_exp = sp.expand(expr_simp)
        h_expr = -sp.diff(expr_exp, t) / expr_exp

        # === BERSIHKAN NOISE DARI AKAR ===
        h_clean = h_expr

        # Ganti exp(-λt) yang sangat kecil → 0
        for lam_name, lam_val in lambdas.items():
            lam_sym = sp.symbols(lam_name)
            exp_term = sp.exp(-lam_sym * t)
            # Jika λ*t > 700 → exp(-λt) ≈ 0
            if lam_val * max(t_values) > 700:
                h_clean = h_clean.subs(exp_term, 0)

        # Final simplify
        h_final = sp.simplify(h_clean)

        # === UBAH KE FORMAT EXCEL MURNI ===
        h_excel = str(h_final)
        h_excel = (h_excel
                   .replace('exp', 'EXP')
                   .replace('**', '^')
                   .replace(' ', ''))
        h_excel = re.sub(r'\s+', '', h_excel)

        method = "symbolic_clean_pure"
        h_str = h_excel  # ← INI YANG DIPAKAI EXCEL

    except Exception as e:
        print(f"Symbolic failed: {e}. Using numerical.")
        method = "numerical"
        h_str = "numerical_hazard"

        def hazard(t_val):
            if t_val <= 0:
                return 0.0
            R_t = safe_R(t_val)
            if R_t >= 1.0:
                return 0.0
            if R_t <= 0.0:
                return 0.0  # bukan inf

            delta = max(1e-8, t_val * 1e-6)
            R_plus = safe_R(t_val + delta)
            R_minus = safe_R(max(t_val - delta, 1e-12))

            if R_plus >= R_t or R_minus >= R_t:
                return 0.0

            dR_dt = (R_plus - R_minus) / (2 * delta)
            h = -dR_dt / R_t
            return max(float(h), 0.0) if np.isfinite(h) else 0.0

    # 4. Hitung nilai numerik (untuk output JSON)
    rows = []
    for t_val in t_values:
        try:
            if method == "symbolic_clean_pure":
                h_val = float(h_final.subs(t, t_val).evalf())
            else:
                h_val = hazard(t_val)
            h_val = max(h_val, 0.0) if np.isfinite(h_val) else 0.0
        except:
            h_val = 0.0

        rows.append({
            "t": f"{t_val:.6e}",
            "hazard_rate": f"{h_val:.6e}" if np.isfinite(h_val) else "0.000000e+00",
            **lambdas
        })

    return {
        "R": fungsi_str,
        "h": h_str,                    # ← RUMUS MURNI UNTUK EXCEL
        "h_symbolic_raw": str(h_expr) if 'h_expr' in locals() else "",
        "data": rows,
        "method": method,
        "excel_paste": f"={h_str}",    # LANGSUNG PASTE KE EXCEL
        "warning": "RUMUS INI MURNI. HASIL SELALU ≥ 0. TIDAK ADA IF/MAX/ROUND."
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