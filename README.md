# PowerSystem_Load_Flow_Analysis
⚡ Power System Analysis Dashboard

Built with: Streamlit • Pandapower • NetworkX • Plotly

🧩 Overview

The Power System Analysis Dashboard is an interactive web app designed to simplify power flow analysis and visualization of electrical networks.
It allows users to create a Pandapower network from CSV files, perform load-flow calculations, and visualize the system as an interactive single-line diagram.

🚀 Run the App
streamlit run powerSystemDashboardApp.py

🛠️ Key Features

✅ CSV Templates Provided — Pre-made Bus and Line CSVs to start quickly.
📂 Upload or Use Sample Data — Flexible input for your own datasets.
⚙️ Automatic Network Building — Creates a Pandapower network from CSV data.
🔁 Load Flow Simulation — Computes voltages, currents, and losses.
📊 Interactive Visualization — Uses NetworkX + Plotly for:

Single-line diagram generation

Color-coded buses (SLACK, PV, PQ)

Line thickness showing power flow

Hover tooltips with detailed values
📉 Comprehensive Results Display — Bus voltages, line flows, and power losses.
💾 Downloadable Outputs — Export Bus and Line results as CSV files.

🧠 Technologies Used
Tool / Library	Purpose
Streamlit	Web app framework for Python
Pandapower	Power flow computation and network modeling
NetworkX	Graph representation of power network
Plotly	Interactive single-line diagram visualization
Pandas / NumPy	Data handling and computation
📘 How to Use

Download the Bus and Line CSV templates from the sidebar.

Fill in your network data or use the provided sample data.

Upload the CSV files in the app.

Click “Run Load Flow” to perform analysis.

View the bus and line results along with the interactive diagram.

Download the computed results as CSVs.

🖼️ Example Output

Bus Results: Voltage magnitude, angle, P/Q injections.

Line Results: Power flows, current, and line losses.

Interactive Diagram:

Color shows bus type

Line width indicates loading

Hover text displays detailed power flow data

⚡ Why This App

This dashboard provides a hands-on, visual approach to understanding load-flow analysis, ideal for:

Students learning Power System Analysis

Engineers running quick network simulations

Educators demonstrating electrical network behavior
