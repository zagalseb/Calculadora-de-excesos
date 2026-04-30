"""
app.py — Calculadora de Exceso de Forja
Streamlit Community Cloud

Estructura del repositorio:
    ├── app.py               ← este archivo
    ├── modelo_excesos.pkl   ← generado por exportar_modelo.py
    └── requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Calculadora de Exceso de Forja",
    page_icon="⚙️",
    layout="centered",
)

# =============================================================================
# CARGA DEL MODELO (con caché para no recargarlo en cada interacción)
# =============================================================================

@st.cache_resource
def cargar_modelo():
    ruta = "modelo_excesos.pkl"
    if not os.path.exists(ruta):
        st.error("❌ No se encontró 'modelo_excesos.pkl'. "
                 "Sube el archivo al repositorio de GitHub.")
        st.stop()
    return joblib.load(ruta)

bundle = cargar_modelo()

modelo           = bundle['modelo']
modelo_q05       = bundle['modelo_q05']
modelo_q95       = bundle['modelo_q95']
encoders_ohe     = bundle['encoders_ohe']
INPUT_COLS       = bundle['INPUT_COLS']
CAT_COLS         = bundle['CAT_COLS']
OUTPUT_COLS      = bundle['OUTPUT_COLS']
FAMILIAS_EXCLUIDAS = bundle['FAMILIAS_EXCLUIDAS']
df_buenas        = bundle['df_buenas_ref']

# =============================================================================
# FUNCIONES AUXILIARES (igual que en el notebook original)
# =============================================================================

def envolvente_minima(de_final: float) -> int:
    if de_final < 1000:
        return 11
    elif de_final <= 2000:
        return 13
    else:
        return 15


def _construir_X(de_final, di_final, altura_final,
                 exceso_de_cfg, exceso_di_cfg, exceso_alt_cfg,
                 peso_cfg, material, tipo, roladora, familia):
    fila_cat = pd.DataFrame([{
        'Material':           material,
        'Tipo':               tipo,
        'Roladora':           roladora,
        'Familia geométrica': familia,
    }])
    cat_enc  = encoders_ohe.transform(fila_cat)
    cat_part = pd.DataFrame(cat_enc,
                            columns=encoders_ohe.get_feature_names_out(CAT_COLS))
    fila_num = pd.DataFrame([{
        'DE Final':                  de_final,
        'DI Final':                  di_final,
        'Altura Final':              altura_final,
        'Exceso DE configurado':     exceso_de_cfg,
        'Exceso DI Configurado':     exceso_di_cfg,
        'Exceso Configurado Altura': exceso_alt_cfg,
        'Peso de Forja Configurado': peso_cfg,
        'Envolvente_Min':            envolvente_minima(de_final),
        'Ratio_DE_DI':               de_final / di_final,
        'Espesor_Pared':             (de_final - di_final) / 2,
    }])
    X = pd.concat([cat_part, fila_num], axis=1)[INPUT_COLS]
    return X


def recomendar_exceso(de_final, di_final, altura_final, peso_cfg,
                      material, tipo, roladora, familia,
                      margen_seguridad=1.0, paso=0.5, max_iter=100):
    env_min = envolvente_minima(de_final)

    exceso_de  = float(env_min + margen_seguridad)
    exceso_di  = float(env_min + margen_seguridad)
    exceso_alt = float(env_min + margen_seguridad)

    cumple_de = cumple_di = cumple_alt = False

    for _ in range(max_iter):
        X = _construir_X(de_final, di_final, altura_final,
                         exceso_de, exceso_di, exceso_alt,
                         peso_cfg, material, tipo, roladora, familia)
        pred_q05 = modelo_q05.predict(X)[0]

        cumple_de  = pred_q05[0] >= env_min
        cumple_di  = pred_q05[1] >= env_min
        cumple_alt = pred_q05[2] >= env_min

        if cumple_de and cumple_di and cumple_alt:
            break

        if not cumple_de:  exceso_de  += paso
        if not cumple_di:  exceso_di  += paso
        if not cumple_alt: exceso_alt += paso

    X_opt    = _construir_X(de_final, di_final, altura_final,
                            exceso_de, exceso_di, exceso_alt,
                            peso_cfg, material, tipo, roladora, familia)
    pred_med = modelo.predict(X_opt)[0]
    pred_q05 = modelo_q05.predict(X_opt)[0]
    pred_q95 = modelo_q95.predict(X_opt)[0]

    mask = (
        (df_buenas['Familia geométrica'] == familia) &
        (df_buenas['Roladora']           == roladora)
    )
    df_similar = df_buenas[mask] if mask.sum() >= 10 else df_buenas

    q05_hist = {
        'DE':  df_similar['DE Exceso Real 1'].quantile(0.05),
        'DI':  df_similar['DI Exceso Real 1'].quantile(0.05),
        'ALT': df_similar['Altura Exceso Real 1'].quantile(0.05),
    }

    resultado = {
        'DE': {
            'exceso_cfg':    round(exceso_de,    1),
            'real_esperado': round(pred_med[0],  1),
            'real_q05':      round(pred_q05[0],  1),
            'real_q95':      round(pred_q95[0],  1),
            'cumple':        cumple_de,
            'q05_hist':      round(q05_hist['DE'], 1),
        },
        'DI': {
            'exceso_cfg':    round(exceso_di,    1),
            'real_esperado': round(pred_med[1],  1),
            'real_q05':      round(pred_q05[1],  1),
            'real_q95':      round(pred_q95[1],  1),
            'cumple':        cumple_di,
            'q05_hist':      round(q05_hist['DI'], 1),
        },
        'ALT': {
            'exceso_cfg':    round(exceso_alt,   1),
            'real_esperado': round(pred_med[2],  1),
            'real_q05':      round(pred_q05[2],  1),
            'real_q95':      round(pred_q95[2],  1),
            'cumple':        cumple_alt,
            'q05_hist':      round(q05_hist['ALT'], 1),
        },
    }

    n_similares = int(mask.sum()) if mask.sum() >= 10 else len(df_buenas)
    return resultado, env_min, n_similares

# =============================================================================
# VALORES VÁLIDOS (para mostrar en la UI)
# =============================================================================

@st.cache_data
def valores_validos():
    categorias = {}
    for i, col in enumerate(CAT_COLS):
        categorias[col] = sorted(encoders_ohe.categories_[i].tolist())
    return categorias

cats = valores_validos()

# =============================================================================
# INTERFAZ
# =============================================================================

st.title("⚙️ Calculadora de Exceso de Forja")
st.markdown("Introduce los parámetros de la pieza para obtener el **exceso mínimo recomendado** a configurar.")

with st.form("calculadora"):

    st.subheader("📋 Características de la pieza")
    col1, col2 = st.columns(2)

    with col1:
        material = st.selectbox("Material", cats['Material'])
        tipo     = st.selectbox("Tipo",     cats['Tipo'])

    with col2:
        roladora = st.selectbox("Roladora",           cats['Roladora'])
        familia  = st.selectbox("Familia geométrica", [
            f for f in cats['Familia geométrica']
            if f not in FAMILIAS_EXCLUIDAS
        ])

    st.subheader("📐 Requerimiento del cliente (mm)")
    col3, col4, col5 = st.columns(3)
    with col3:
        de_final     = st.number_input("DE Final (mm)",     min_value=1.0, value=200.0, step=1.0)
    with col4:
        di_final     = st.number_input("DI Final (mm)",     min_value=1.0, value=100.0, step=1.0)
    with col5:
        altura_final = st.number_input("Altura Final (mm)", min_value=1.0, value=80.0,  step=1.0)

    st.subheader("⚙️ Parámetros de proceso")
    col6, col7 = st.columns(2)
    with col6:
        peso_cfg = st.number_input("Peso configurado (kg)", min_value=0.0, value=50.0, step=1.0)
    with col7:
        margen   = st.slider("Margen de seguridad extra (mm)", min_value=0.5, max_value=5.0,
                             value=1.0, step=0.5)

    calcular = st.form_submit_button("🔍 Calcular configuración", use_container_width=True)

# =============================================================================
# RESULTADO
# =============================================================================

if calcular:
    # --- Validaciones ---
    errores = []
    if de_final <= di_final:
        errores.append("DE Final debe ser mayor que DI Final.")
    if de_final <= 0 or di_final <= 0 or altura_final <= 0:
        errores.append("DE Final, DI Final y Altura Final deben ser > 0.")
    if margen < 1.5:
        st.info(f"💡 Margen extra bajo ({margen} mm). Considera subirlo para mayor holgura.")

    if errores:
        for e in errores:
            st.error(f"⚠️ {e}")
        st.stop()

    with st.spinner("Calculando exceso óptimo..."):
        resultado, env_min, n_similares = recomendar_exceso(
            de_final=de_final, di_final=di_final, altura_final=altura_final,
            peso_cfg=peso_cfg, material=material, tipo=tipo,
            roladora=roladora, familia=familia,
            margen_seguridad=margen,
        )

    # --- Encabezado del resultado ---
    todos_cumplen = all(r['cumple'] for r in resultado.values())

    st.divider()
    st.subheader("📊 Exceso recomendado a configurar")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Mínimo de seguridad", f"{env_min} mm")
    col_info2.metric("Intervalo de confianza", "90%  (Q5 – Q95)")
    col_info3.metric("Piezas de referencia", f"{n_similares}")

    # --- Tarjetas por dimensión ---
    etiquetas = {
        'DE':  ('Diámetro Exterior', de_final),
        'DI':  ('Diámetro Interior', di_final),
        'ALT': ('Altura',            altura_final),
    }

    for dim, (nombre, dim_final) in etiquetas.items():
        r = resultado[dim]
        icono = "✅" if r['cumple'] else "⚠️"

        with st.container(border=True):
            st.markdown(f"#### {icono} {nombre} — Final pedido: **{dim_final} mm**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Exceso a configurar",      f"{r['exceso_cfg']:.2f} mm")
            c2.metric("Real esperado",            f"{r['real_esperado']:.2f} mm")
            c3.metric("Rango de Exceso estimado", f"[{r['real_q05']:.2f} – {r['real_q95']:.2f}] mm")

            cumple_min = r['real_q05'] >= env_min
            if cumple_min:
                st.success(f"Q5 garantizado: {r['real_q05']:.2f} mm ≥ {env_min} mm ✅")
            else:
                st.warning(
                    f"Q5 garantizado: {r['real_q05']:.2f} mm < {env_min} mm — "
                    f"Alta variabilidad. Considera aumentar el margen o revisar el proceso."
                )

    # --- Veredicto global ---
    st.divider()
    if todos_cumplen:
        st.success("✅ TODAS LAS RESTRICCIONES CUMPLIDAS")
    else:
        st.warning("⚠️ ALGUNAS DIMENSIONES CON ALTA VARIABILIDAD — VER DETALLE ARRIBA")
