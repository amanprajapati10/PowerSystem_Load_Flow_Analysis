"""
Power System Analysis Dashboard (Streamlit + pandapower + NetworkX + Plotly)

Run:
  streamlit run powerSystemDashboardApp.py

This app:
 - Provides CSV templates for Bus and Line data
 - Lets user upload CSVs or use sample data
 - Builds pandapower network, runs power flow
 - Displays results (bus voltages, line flows, losses)
 - Draws an interactive single-line diagram using NetworkX + Plotly

"""

import io
import pandas as pd
import numpy as np
import streamlit as st

# Heavy imports with fallback
try:
    import pandapower as pp
    import networkx as nx
    import plotly.graph_objects as go
except Exception:
    st.set_page_config(page_title="Power System Dashboard", layout="wide")
    st.title("Power System Dashboard")
    st.error("Missing packages. Install with:\n\npip install streamlit pandas pandapower networkx plotly")
    st.stop()

st.set_page_config(page_title="Power System Analysis Dashboard", layout="wide")
st.title("⚡ Power System Analysis Dashboard")

# Sidebar: instructions and sample files
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1. Prepare **Bus data** and **Line data** CSV files using the templates.
        2. Upload both files or use sample data.
        3. Click **Run Load Flow** to see results.
        """
    )
    st.markdown("---")

# -------------------------------
# SAMPLE DATA
# -------------------------------
sample_bus_csv = (
    "bus_id,bus_name,bus_type,vm_pu,va_degree,p_gen_mw,q_gen_mvar,p_load_mw,q_load_mvar\n"
    "1,Bus1,SLACK,1.05,0,409,189,0,0\n"
    "2,Bus2,PQ,0.98,,0,0,256.6,110.2\n"
    "3,Bus3,PQ,1.00,,0,0,138.6,45.2\n"
)

sample_line_csv = (
    "from_bus,to_bus,length_km,r_ohm_per_km,x_ohm_per_km,c_nf_per_km,max_i_ka,line_name\n"
    "1,2,1,0.02,0.06,0,0.3,Line_1_2\n"
    "1,3,1,0.08,0.24,0,0.25,Line_1_3\n"
    "2,3,1,0.06,0.18,0,0.2,Line_2_3\n"
)

st.sidebar.download_button("Download sample Bus CSV", data=sample_bus_csv, file_name="bus_sample.csv")
st.sidebar.download_button("Download sample Line CSV", data=sample_line_csv, file_name="line_sample.csv")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.subheader("Upload CSV Data")
col1, col2 = st.columns(2)

with col1:
    bus_file = st.file_uploader("Upload Bus CSV", type=["csv"])
with col2:
    line_file = st.file_uploader("Upload Line CSV", type=["csv"])

use_sample = st.checkbox("Use sample data instead", value=True)

if use_sample:
    bus_df = pd.read_csv(io.StringIO(sample_bus_csv))
    line_df = pd.read_csv(io.StringIO(sample_line_csv))
else:
    if bus_file is None or line_file is None:
        st.warning("Upload both Bus and Line CSVs or select 'Use sample data'.")
        st.stop()
    bus_df = pd.read_csv(bus_file)
    line_df = pd.read_csv(line_file)

st.markdown("### Bus Data")
st.dataframe(bus_df)
st.markdown("### Line Data")
st.dataframe(line_df)

# -------------------------------
# BUTTON: RUN LOAD FLOW
# -------------------------------
if st.button("Run Load Flow"):
    net = pp.create_empty_network()

    # Map bus ids
    bus_index_map = {}
    for _, row in bus_df.iterrows():
        bus_idx = pp.create_bus(net, vn_kv=132, name=row["bus_name"])
        bus_index_map[int(row["bus_id"])] = bus_idx

    # Add generators, slack, loads
    for _, row in bus_df.iterrows():
        bus_idx = bus_index_map[int(row["bus_id"])]
        btype = row["bus_type"].upper()
        p_gen, q_gen = row["p_gen_mw"], row["q_gen_mvar"]
        p_load, q_load = row["p_load_mw"], row["q_load_mvar"]
        vm_pu = row["vm_pu"] if not pd.isna(row["vm_pu"]) else 1.0

        if btype == "SLACK":
            pp.create_ext_grid(net, bus=bus_idx, vm_pu=vm_pu, name=f"Slack_{row['bus_name']}")
        elif p_gen > 0 or q_gen > 0:
            pp.create_gen(net, bus=bus_idx, p_mw=p_gen, vm_pu=vm_pu, name=f"Gen_{row['bus_name']}")

        if p_load > 0 or q_load > 0:
            pp.create_load(net, bus=bus_idx, p_mw=p_load, q_mvar=q_load, name=f"Load_{row['bus_name']}")

    # Add lines
    for _, row in line_df.iterrows():
        from_idx = bus_index_map[int(row["from_bus"])]
        to_idx = bus_index_map[int(row["to_bus"])]
        pp.create_line_from_parameters(
            net, from_idx, to_idx, length_km=row["length_km"],
            r_ohm_per_km=row["r_ohm_per_km"], x_ohm_per_km=row["x_ohm_per_km"],
            c_nf_per_km=row["c_nf_per_km"], max_i_ka=row["max_i_ka"],
            name=row["line_name"]
        )

    # Run power flow
    try:
        pp.runpp(net)
    except Exception as e:
        st.error(f"Power Flow Error: {e}")
        st.stop()

    # -------------------------------
    # SHOW RESULTS
    # -------------------------------
    st.subheader("Bus Results")
    st.dataframe(net.res_bus)

    st.subheader("Line Results")
    st.dataframe(net.res_line)

    total_p_loss = round(net.res_line["pl_mw"].sum(), 3)
    total_q_loss = round(net.res_line["ql_mvar"].sum(), 3)

    st.markdown(f"**Total Active Power Loss:** {total_p_loss} MW")
    st.markdown(f"**Total Reactive Power Loss:** {total_q_loss} MVAR")

    # -------------------------------
    # INTERACTIVE PLOT
    # -------------------------------
    st.subheader("Interactive Single-Line Diagram")

    # NetworkX graph
    G = nx.Graph()
    for _, row in line_df.iterrows():
        G.add_edge(row["from_bus"], row["to_bus"])

    pos = nx.spring_layout(G, seed=42)

    # Colors for bus types
    bus_colors = {"SLACK": "green", "PV": "blue", "PQ": "red"}

    fig = go.Figure()

    # Draw lines
    for idx, row in net.res_line.iterrows():
        from_bus = line_df.iloc[idx]["from_bus"]
        to_bus = line_df.iloc[idx]["to_bus"]
        x0, y0 = pos[from_bus]
        x1, y1 = pos[to_bus]

        line_width = max(1, abs(row["p_from_mw"]) / 10)

        # Add line
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="black", width=line_width),
            hoverinfo="none"
        ))

        # Add loss text
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            text=[f"Loss: {row['pl_mw']:.2f} MW / {row['ql_mvar']:.2f} MVAR"],
            mode="text", showlegend=False
        ))

    # Draw buses
    for _, row in bus_df.iterrows():
        b_id = row["bus_id"]
        x, y = pos[b_id]
        btype = row["bus_type"].upper()
        color = bus_colors.get(btype, "gray")

        res = net.res_bus.loc[bus_index_map[b_id]]
        hover_text = (
            f"Bus {row['bus_name']}<br>"
            f"Type: {btype}<br>"
            f"V = {res['vm_pu']:.3f} pu<br>"
            f"Angle = {res['va_degree']:.2f}°<br>"
            f"Pgen = {row['p_gen_mw']} MW<br>"
            f"Qgen = {row['q_gen_mvar']} MVAR<br>"
            f"Pload = {row['p_load_mw']} MW<br>"
            f"Qload = {row['q_load_mvar']} MVAR"
        )

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=18, color=color),
            text=[f"Bus {b_id}"], textposition="top center",
            hovertext=hover_text, hoverinfo="text"
        ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="#485000",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # DOWNLOAD BUTTONS
    # -------------------------------
    st.subheader("Download Results")
    bus_csv = net.res_bus.to_csv(index=True)
    line_csv = net.res_line.to_csv(index=True)
    st.download_button("Download Bus Results CSV", data=bus_csv, file_name="bus_results.csv")
    st.download_button("Download Line Results CSV", data=line_csv, file_name="line_results.csv")

else:
    st.info("Click **Run Load Flow** to execute the power flow.")
