# SeismicForward: 1D Seismic Forward Modelling

This project provides a Python-based forward model for **1D seismic wave propagation in layered media** and a **web interface** for interactively building:

- Layered earth models (thickness, velocity, density)
- Source pulse (Ricker wavelet)
- Receiver measurement locations (offsets)

The app then simulates synthetic seismic traces and displays a shot gather.

## Features

- Acoustic impedance reflection coefficients at layer interfaces
- Hyperbolic moveout approximation across receiver offsets
- Time-domain synthetic traces by convolving reflectivity spikes with a pulse
- Interactive Streamlit UI for rapid experimentation

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project files

- `forward_model.py`: Core modelling functions and data classes
- `streamlit_app.py`: Interactive web interface
- `requirements.txt`: Dependencies
