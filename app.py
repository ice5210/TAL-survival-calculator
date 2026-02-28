import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. Core Numerical Logic (TAL Formulas) ---
class EasyDict(dict):
    def __getattr__(self, name): return self.get(name, 0)
    def __setattr__(self, name, value): self[name] = value

def TAL_CRIT_Formulas(crit, end):
    if crit > end:
        x = crit - end
        return x / (x + 1000) # Crit Chance
    elif end > crit:
        x = end - crit
        return -(x / (x + 1000)) # Glancing Chance (Negative)
    return 0

def TAL_HAC_Formulas(hac, hac_eva):
    if hac > hac_eva:
        x = hac - hac_eva
        return x / (x + 1000)
    return 0 

def TAL_SDR_Formulas(hp=31742, def_=3600, sdr=1108, crit_dmg_extra=0.6, crit_dmg_res=0.453, crit_multi=2.5, 
                     crit_ch=3200, end=3438, heavy_dmg_extra=1.7, heavy_dmg_res=0.0, heavy_ch=2500, heavy_ch_eva=893, 
                     shield_bk_ch=1.198, shield_bk_pen_ch=0.742, shield_bk_red=0.425):
    
    dr_dict = EasyDict(hp=hp)
    effective_hp = float(hp)
    
    # Base Mitigation: Defense & SDR
    def_dr = def_ / ((def_ + 2500) / 100) if def_ > 0 else 0
    base_mitigated_hp = effective_hp / max((1 - (def_dr / 100)), 0.0001)
    sdr_dr = sdr / ((sdr + 700) / 100) if sdr > 0 else 0
    base_mitigated_hp /= max((1 - (sdr_dr / 100)), 0.0001)
    
    # Multiplier Logic
    net_crit_dmg = max(crit_dmg_extra - crit_dmg_res, 0.0)
    crit_dmg_full = max(crit_multi, 0.1) * (1 + net_crit_dmg)
    h_dmg_full = 1 + max(heavy_dmg_extra - heavy_dmg_res, 0.5)
    
    # --- Dynamic Max Hit (Survival Cap) with JUMP Logic ---
    max_hit_hp = base_mitigated_hp
    
    # JUMP 1: Endurance vs Crit (Glancing Immunity)
    if crit_ch > end:
        max_hit_hp /= crit_dmg_full
    
    # JUMP 2: HAC vs EVA (Heavy Immunity)
    if heavy_ch > heavy_ch_eva:
        max_hit_hp /= h_dmg_full
        
    # JUMP 3: 100% Block (Block - Pen >= 1.0)
    net_block_ch = shield_bk_ch - shield_bk_pen_ch
    if net_block_ch >= 1.0:
        max_hit_hp /= (1 - shield_bk_red)
    
    # --- EHP (Average Expectation) ---
    crit_res_val = TAL_CRIT_Formulas(crit=crit_ch, end=end)
    eff_hp_base = (base_mitigated_hp + (base_mitigated_hp / max(crit_multi, 0.1))) / 2
    
    if crit_res_val > 0: # Crit Vulnerable
        avg_hp = (eff_hp_base * (1 - crit_res_val)) + ((base_mitigated_hp / crit_dmg_full) * crit_res_val)
    else: # Glancing Active
        glancing_ch = abs(crit_res_val)
        avg_hp = (eff_hp_base * (1 - glancing_ch)) + (base_mitigated_hp * glancing_ch)

    h_ch_weva = TAL_HAC_Formulas(hac=heavy_ch, hac_eva=heavy_ch_eva)
    avg_hp = (avg_hp * (1 - h_ch_weva)) + ((avg_hp / h_dmg_full) * h_ch_weva)

    if net_block_ch > 0:
        actual_block_ch = min(net_block_ch, 1.0)
        avg_hp = (avg_hp * (1 - actual_block_ch)) + ((avg_hp / (1 - shield_bk_red)) * actual_block_ch)

    dr_dict.final_effective_hp = avg_hp
    dr_dict.final_max_hit_hp = max_hit_hp
    return dr_dict

# --- 2. UI Layout ---
st.set_page_config(page_title="TAL survival calculator", layout="wide")
st.title("TAL survival calculator")

with st.sidebar:
    st.header("👤 Your Stats")
    in_hp = st.number_input("Max HP", value=31742.0)
    in_def = st.number_input("Defense", value=3600.0)
    in_sdr = st.number_input("SDR", value=1108.0)
    in_end = st.number_input("Endurance", value=3438.0)
    in_crit_res = st.number_input("Crit DMG Res", value=0.453)
    in_heavy_eva = st.number_input("Heavy EVA", value=893.0)
    in_heavy_res = st.number_input("Heavy DMG Res", value=0.0)
    in_sbk_ch = st.number_input("Shield Block Chance", value=1.198, format="%.3f")
    in_sbk_red = st.number_input("Shield Reduction", value=0.425)

    st.header("👹 Enemy Stats")
    in_crit_ch = st.number_input("Enemy Crit Stat", value=3200.0)
    in_multi = st.number_input("Crit Multiplier", value=2.5)
    in_crit_dmg_extra = st.number_input("Enemy Extra Crit DMG", value=0.6)
    in_heavy_ch = st.number_input("Enemy Heavy (HAC)", value=2500.0)
    in_heavy_dmg_extra = st.number_input("Enemy Extra Heavy DMG", value=1.7)
    in_sbk_pen = st.number_input("Enemy Shield Pen", value=0.742)

# Calculate Base
base_params = dict(hp=in_hp, def_=in_def, sdr=in_sdr, crit_dmg_res=in_crit_res, crit_multi=in_multi, crit_ch=in_crit_ch, end=in_end, heavy_dmg_res=in_heavy_res, heavy_ch=in_heavy_ch, heavy_ch_eva=in_heavy_eva, crit_dmg_extra=in_crit_dmg_extra, heavy_dmg_extra=in_heavy_dmg_extra, shield_bk_ch=in_sbk_ch, shield_bk_pen_ch=in_sbk_pen, shield_bk_red=in_sbk_red)
res = TAL_SDR_Formulas(**base_params)

c1, c2, c3 = st.columns(3)
c1.metric("Average EHP", f"{res.final_effective_hp:,.0f}")
c2.metric("Max Hit Capacity", f"{res.final_max_hit_hp:,.0f}")
c3.metric("Status", "Glancing Active" if in_end > in_crit_ch else "Crit Vulnerable")

# --- 3. Efficiency Comparison with User Increments ---
st.write("---")
st.subheader("📊 Stat Efficiency Comparison (Gain from Custom Boost)")

with st.expander("⚙️ Individual Stat Increments for Comparison"):
    col1, col2, col3 = st.columns(3)
    inc_hp = col1.number_input("HP Boost Amount", value=1000.0, step=100.0)
    inc_def = col2.number_input("DEF Boost Amount", value=100.0, step=10.0)
    inc_sdr = col3.number_input("SDR Boost Amount", value=100.0, step=10.0)
    col4, col5, col6 = st.columns(3)
    inc_end = col4.number_input("Endurance Boost Amount", value=100.0, step=10.0)
    inc_eva = col5.number_input("Heavy EVA Boost Amount", value=100.0, step=10.0)
    inc_sbk = col6.number_input("Shield Block % Boost Amount", value=0.100, step=0.010, format="%.3f")

tests = [
    {"label": f"HP +{int(inc_hp)}", "params": {**base_params, "hp": in_hp + inc_hp}},
    {"label": f"DEF +{int(inc_def)}", "params": {**base_params, "def_": in_def + inc_def}},
    {"label": f"SDR +{int(inc_sdr)}", "params": {**base_params, "sdr": in_sdr + inc_sdr}},
    {"label": f"END +{int(inc_end)}", "params": {**base_params, "end": in_end + inc_end}},
    {"label": f"HVY EVA +{int(inc_eva)}", "params": {**base_params, "heavy_ch_eva": in_heavy_eva + inc_eva}},
    {"label": f"SBK +{inc_sbk:.2f}", "params": {**base_params, "shield_bk_ch": in_sbk_ch + inc_sbk}},
]

eff_results = []
for t in tests:
    sim = TAL_SDR_Formulas(**t["params"])
    ehp_g = ((sim.final_effective_hp / res.final_effective_hp) - 1) * 100
    mhp_g = ((sim.final_max_hit_hp / res.final_max_hit_hp) - 1) * 100
    eff_results.append({"Comparison": t["label"], "Gain (%)": ehp_g, "Type": "Avg EHP Gain"})
    eff_results.append({"Comparison": t["label"], "Gain (%)": mhp_g, "Type": "Max Hit Gain"})

fig_eff = px.bar(pd.DataFrame(eff_results), x="Comparison", y="Gain (%)", color="Type", barmode="group",
                 color_discrete_map={"Avg EHP Gain": "#00FFCC", "Max Hit Gain": "#FF4B4B"},
                 template="plotly_dark")
st.plotly_chart(fig_eff, use_container_width=True)

# --- 4. Dynamic Plotting (Mid-Point Analysis) ---
st.write("---")
target_label = st.selectbox("Simulate Growth (Centered on Current):", ["Defense", "Endurance", "SDR", "Shield Block Chance", "Max HP", "Heavy EVA"])
stat_mapping = {"Defense": "def_", "Endurance": "end", "SDR": "sdr", "Shield Block Chance": "shield_bk_ch", "Max HP": "hp", "Heavy EVA": "heavy_ch_eva"}
current_val = base_params[stat_mapping[target_label]]

# Mid-point Logic
if target_label in ["Shield Block Chance", "Heavy EVA", "Endurance"]:
    test_range = np.linspace(max(0, current_val - 1.0 if "Chance" in target_label else current_val - 2000), current_val + 1.0 if "Chance" in target_label else current_val + 2000, 300)
else:
    test_range = np.linspace(current_val * 0.5, current_val * 1.5, 200)

y_ehp, y_max_hit = [], []
for v in test_range:
    p = base_params.copy()
    p[stat_mapping[target_label]] = v
    sim = TAL_SDR_Formulas(**p)
    y_ehp.append(sim.final_effective_hp)
    y_max_hit.append(sim.final_max_hit_hp)

df_plot = pd.DataFrame({target_label: test_range, 'Average EHP': y_ehp, 'Max Hit': y_max_hit})
fig = px.line(df_plot, x=target_label, y=['Average EHP', 'Max Hit'], color_discrete_map={'Average EHP': '#00FFCC', 'Max Hit': '#FF4B4B'}, template="plotly_dark")
fig.add_scatter(x=[current_val], y=[res.final_effective_hp], name="Current EHP", marker=dict(size=12, color='#00FFCC'))
fig.add_scatter(x=[current_val], y=[res.final_max_hit_hp], name="Current Max Hit", marker=dict(size=12, color='#FF4B4B'))

# Visual Jumps
if target_label == "Endurance": fig.add_vline(x=in_crit_ch, line_dash="dash", line_color="orange", annotation_text="Glancing Jump")
elif target_label == "Shield Block Chance": fig.add_vline(x=in_sbk_pen + 1.0, line_dash="dash", line_color="orange", annotation_text="100% Block Jump")
elif target_label == "Heavy EVA": fig.add_vline(x=in_heavy_ch, line_dash="dash", line_color="orange", annotation_text="Heavy Immunity")

st.plotly_chart(fig, use_container_width=True)


