from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forward_model import LayerModel, Pulse, synthesize_gather

st.set_page_config(page_title="1D Seismic Forward Modelling", layout="wide")
st.title("1D Seismic Forward Modelling in Layered Media")
st.caption("Build a layered model, define a source pulse, and simulate recordings at receiver offsets.")

with st.sidebar:
    st.header("Acquisition Settings")
    n_receivers = st.slider("Number of receivers", 1, 64, 24)
    max_offset = st.number_input("Maximum receiver offset (m)", min_value=1.0, value=2000.0, step=50.0)
    dt = st.number_input("Sampling interval dt (s)", min_value=0.0005, value=0.002, step=0.0005, format="%.4f")
    tmax = st.number_input("Record length (s)", min_value=0.1, value=2.0, step=0.1)

    st.header("Pulse")
    dominant_frequency = st.number_input("Dominant frequency (Hz)", min_value=1.0, value=25.0, step=1.0)
    phase = st.number_input("Pulse phase/time shift (s)", value=0.0, step=0.001, format="%.3f")

st.subheader("Layered Model")
default_layers = pd.DataFrame(
    {
        "thickness_m": [300.0, 500.0, 700.0, 900.0],
        "velocity_m_s": [1600.0, 2200.0, 2800.0, 3400.0],
        "density_kg_m3": [1900.0, 2100.0, 2300.0, 2450.0],
    }
)

layers = st.data_editor(
    default_layers,
    num_rows="dynamic",
    use_container_width=True,
    key="layers",
)

run = st.button("Run Forward Modelling", type="primary")

if run:
    try:
        model = LayerModel.from_iterables(
            layers["thickness_m"].to_numpy(),
            layers["velocity_m_s"].to_numpy(),
            layers["density_kg_m3"].to_numpy(),
        )
        pulse = Pulse(dominant_frequency=dominant_frequency, phase=phase)
        receivers = np.linspace(0, max_offset, n_receivers)
        t, data = synthesize_gather(model, pulse, receivers=receivers, dt=dt, tmax=tmax)

        st.success("Forward modelling completed.")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Example Trace")
            trace_idx = st.slider("Receiver index", 0, len(receivers) - 1, min(5, len(receivers) - 1))
            fig_trace = go.Figure()
            fig_trace.add_trace(
                go.Scatter(x=t, y=data[trace_idx], mode="lines", name=f"Offset={receivers[trace_idx]:.1f} m")
            )
            fig_trace.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
            st.plotly_chart(fig_trace, use_container_width=True)

        with col2:
            st.markdown("#### Synthetic Shot Gather")
            df = pd.DataFrame(data, index=np.round(receivers, 2), columns=np.round(t, 4))
            fig_img = px.imshow(
                df,
                aspect="auto",
                labels=dict(x="Time (s)", y="Offset (m)", color="Amplitude"),
                origin="lower",
                color_continuous_scale="RdBu_r",
            )
            fig_img.update_layout(coloraxis_showscale=True)
            st.plotly_chart(fig_img, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not run model: {exc}")

with st.expander("How the model works"):
    st.write(
        """
        - Reflection coefficients are computed at layer interfaces using acoustic impedance contrasts.
        - Arrival times are approximated with normal-moveout hyperbolas for each offset.
        - A Ricker pulse is convolved with sparse reflectivity spikes to generate synthetic traces.
        """
    )
