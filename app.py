import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta

# ==========================================
# 1. SETUP: DATA GENERATION & LOADING
# ==========================================
@st.cache_data
def load_data():
    # --- A. Generate Parts DB (Simulated) ---
    data = {
        'Part Name': [
            'Engine Oil', 'Oil Filter', 'Air Filter', 'Fuel Filter',
            'Brake Pads Front', 'Brake Pads Rear', 'Brake Fluid',
            'Spark Plug', 'Transmission Oil',
            'Clutch Plate', 'Chain Set', 'Battery', 'Headlight Bulb',
            'Clutch Cable', 'Accelerator Cable', 'Brake Cable',
            'Fork Oil Seal', 'Engine Gasket', 'Piston Rings', 'Valve Set'
        ],
        'Category': [
            'Fluids', 'Fluids', 'Fluids', 'Fluids',
            'Braking', 'Braking', 'Fluids',
            'Electrical', 'Fluids',
            'Transmission', 'Transmission', 'Electrical', 'Electrical',
            'Cables', 'Cables', 'Cables',
            'Sealing', 'Sealing', 'Engine', 'Engine'
        ],
        'Range (KM)': [
            3000, 3000, 10000, 15000,
            15000, 20000, 20000,
            12000, 20000,
            35000, 25000, 40000, 50000,
            25000, 25000, 25000,
            30000, 60000, 60000, 60000
        ],
        'Time Limit (Months)': [
            6, 6, 12, 12,
            18, 24, 24,
            12, 24,
            48, 30, 36, 60,
            48, 48, 48,
            48, 72, 72, 72
        ],
        'Is_Critical': [
            1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0
        ]
    }
    df_parts = pd.DataFrame(data)

    # --- B. Load History File ---
    # NOTE: In a real hosted app, you would upload your 'data orginal.csv' 
    # to the GitHub repository or use a file uploader in the app.
    # For this example, we assume the file is in the same folder as app.py
    
    hist_path = 'data orginal.csv' # Ensure this file exists in your folder!
    
    if not os.path.exists(hist_path):
        return None, None
        
    try:
        try: df_hist = pd.read_csv(hist_path, low_memory=False)
        except: df_hist = pd.read_csv(hist_path, encoding='latin1', low_memory=False)
        
        # Clean Columns
        cols = {c.lower(): c for c in df_hist.columns}
        col_reg = next((cols[c] for c in cols if 'reg' in c or 'licence' in c or 'veh' in c), None)
        col_odo = next((cols[c] for c in cols if 'odo' in c or 'km' in c), None)
        col_date = next((cols[c] for c in cols if 'date' in c or 'dt' in c), None)
        col_desc = next((cols[c] for c in cols if 'desc' in c or 'part' in c), None)
        col_model = next((cols[c] for c in cols if 'model' in c or 'variant' in c), None)
        col_city = next((cols[c] for c in cols if 'city' in c or 'loc' in c), None)

        if not col_reg: return df_parts, None

        # Clean Data
        df_hist[col_odo] = df_hist[col_odo].astype(str).str.replace(',', '', regex=False)
        df_hist[col_odo] = pd.to_numeric(df_hist[col_odo], errors='coerce').fillna(0)
        df_hist[col_date] = pd.to_datetime(df_hist[col_date], errors='coerce')
        
        # Store column names in a dict to return
        cols_map = {
            'reg': col_reg, 'odo': col_odo, 'date': col_date,
            'desc': col_desc, 'model': col_model, 'city': col_city
        }
        
        return df_parts, df_hist, cols_map
        
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 2. LOGIC ENGINE
# ==========================================
def analyze_vehicle(reg_no, current_odo, df_parts, df_hist, cols):
    reg_no = reg_no.strip().upper()
    
    # regex check
    if not re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', reg_no):
        return "ERROR", "Invalid Registration Format. Use format like AB12CD1234."

    # Filter History
    df_v = df_hist[df_hist[cols['reg']] == reg_no].copy()
    
    if df_v.empty:
        return "ERROR", f"Vehicle '{reg_no}' not found in history."

    # Get Last Valid Odo
    df_valid = df_v[df_v[cols['odo']] > 0]
    if not df_valid.empty:
        last_rec = df_valid.sort_values(by=[cols['date'], cols['odo']], ascending=[False, False]).iloc[0]
        last_odo = last_rec[cols['odo']]
        last_date = last_rec[cols['date']]
    else:
        last_odo = 0
        last_date = df_v.sort_values(by=cols['date'], ascending=False).iloc[0][cols['date']]

    if current_odo < last_odo:
        return "CRITICAL", f"Current Odometer ({int(current_odo)}) cannot be less than Last Service Odometer ({int(last_odo)})."

    # Metadata
    latest = df_v.sort_values(by=cols['date'], ascending=False).iloc[0]
    model = latest[cols['model']] if cols['model'] else "Unknown"
    city = latest[cols['city']] if cols['city'] else "Unknown"

    # Analysis Loop
    maintenance_list = []
    cat_scores = {k: {'score': 0, 'count': 0} for k in ['Fluids', 'Transmission', 'Sealing', 'Electrical', 'Braking', 'Cables', 'Engine']}
    current_date = datetime.now()

    for _, item in df_parts.iterrows():
        p_name = item['Part Name']
        p_cat = item['Category']
        p_km = item['Range (KM)']
        p_time = item['Time Limit (Months)'] * 30
        is_crit = item['Is_Critical']

        # Find last replacement
        # Simple keyword match
        keywords = set(str(p_name).lower().split())
        
        matches = []
        for _, row in df_v.iterrows():
            desc = str(row[cols['desc']]).lower()
            if any(k in desc for k in keywords if len(k) > 3):
                matches.append(row)
        
        part_last_odo = 0
        part_last_date = None
        
        if matches:
            df_m = pd.DataFrame(matches)
            df_m = df_m[df_m[cols['odo']] > 0]
            if not df_m.empty:
                best = df_m.sort_values(by=cols['date'], ascending=False).iloc[0]
                part_last_odo = best[cols['odo']]
                part_last_date = best[cols['date']]

        # Calc
        dist_run = current_odo - part_last_odo
        km_health = max(0, 1 - (dist_run / p_km))
        
        time_health = 1.0
        due_date = "N/A"
        
        if pd.notnull(part_last_date):
            days_run = (current_date - part_last_date).days
            time_health = max(0, 1 - (days_run / p_time))
            due_date = (part_last_date + timedelta(days=p_time)).strftime('%Y-%m-%d')
        elif current_odo > p_km:
             due_date = "Immediate"

        final = min(km_health, time_health)

        # Aggregate
        if p_cat in cat_scores:
            cat_scores[p_cat]['score'] += final
            cat_scores[p_cat]['count'] += 1

        if final < 0.2:
            reason = f"Overdue {int(dist_run - p_km)}km" if km_health < time_health else "Time Expired"
            maintenance_list.append({
                'Part': p_name,
                'Status': 'REPLACE' if final == 0 else 'WARNING',
                'Health': f"{int(final*100)}%",
                'Due': due_date,
                'Reason': reason,
                'Critical': is_crit
            })

    return "SUCCESS", {
        'meta': {'reg': reg_no, 'model': model, 'city': city, 'last_date': last_date, 'last_odo': last_odo},
        'scores': cat_scores,
        'action': maintenance_list
    }

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Vehicle Health Pro", page_icon="🚗")

st.title("🚗 Vehicle Health Scorecard")

# Load Data
data_load_state = st.text('Loading data...')
df_parts, df_hist, cols = load_data()
data_load_state.text('')

if df_hist is None:
    st.error("❌ History File not found! Please upload 'data orginal.csv' to the app directory.")
    st.stop()

# INPUTS
with st.form("check_form"):
    col1, col2 = st.columns(2)
    with col1:
        u_reg = st.text_input("Registration Number", placeholder="AB12CD1234").upper()
    with col2:
        u_odo = st.number_input("Current Odometer", min_value=0, step=100)
    
    submitted = st.form_submit_button("Analyze Health")

if submitted:
    status, result = analyze_vehicle(u_reg, u_odo, df_parts, df_hist, cols)
    
    if status == "ERROR":
        st.error(result)
    elif status == "CRITICAL":
        st.warning(result)
    else:
        # DISPLAY RESULTS
        meta = result['meta']
        st.success(f"Vehicle Found: {meta['model']} ({meta['city']})")
        
        # Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Last Service Date", meta['last_date'].strftime('%Y-%m-%d'))
        m2.metric("Last Odometer", f"{int(meta['last_odo']):,} km")
        m3.metric("Km Driven Since", f"{int(u_odo - meta['last_odo']):,} km")
        
        st.divider()
        
        # Overall Score
        total_score = 0
        total_weight = 0
        weights = {"Fluids": 0.28, "Transmission": 0.20, "Sealing": 0.12, "Electrical": 0.10, "Braking": 0.18, "Cables": 0.08, "Engine": 0.04}
        
        for cat, data in result['scores'].items():
            if data['count'] > 0:
                s = (data['score'] / data['count']) * 100
                total_score += s * weights.get(cat, 0)
                total_weight += weights.get(cat, 0)
        
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        st.subheader(f"Overall Health: {final_score:.1f}%")
        st.progress(int(final_score))
        
        # Action Items
        items = result['action']
        if items:
            st.subheader(f"🛑 Action Required ({len(items)} Items)")
            
            # Sort critical first
            items.sort(key=lambda x: x['Critical'], reverse=True)
            
            for item in items:
                if item['Critical']:
                    st.error(f"**{item['Part']}** | Due: {item['Due']} | {item['Reason']}")
                else:
                    st.info(f"**{item['Part']}** | Due: {item['Due']} | {item['Reason']}")
        else:
            st.balloons()
            st.success("✅ Vehicle is in excellent condition!")
