# app.py
import json
from dataclasses import asdict, is_dataclass

import streamlit as st
import plotly.graph_objects as go

import PlaneStress as Ps


#
# Params <-> Dict conversion
#
PARAM_FIELDS = [
    "w", "h", "a", "b", "b_end",
    "E", "nu", "t", "t_end",
    "q",
    "el_size_factor",
    "param_steps",
    "param_b",
    "param_t",
]


def modelparams_to_dict(mp: object) -> dict:
    d = {}
    for k in PARAM_FIELDS:
        if hasattr(mp, k):
            d[k] = getattr(mp, k)
    return d


def dict_to_modelparams(d: dict, mp: object) -> object:
    for k, v in d.items():
        if hasattr(mp, k):
            try:
                setattr(mp, k, v)
            except Exception:
                pass
    return mp


def ui_values_to_params_dict(
    w, h, a, b, b_end,
    E, nu, t, t_end,
    q,
    el_slider_value,
    param_steps,
    vary_b,
    vary_t,
) -> dict:
    out = {}

    def put(name, val):
        if val is not None:
            out[name] = val

    put("w", w)
    put("h", h)
    put("a", a)
    put("b", b)
    put("b_end", b_end)
    put("E", E)
    put("nu", nu)
    put("t", t)
    put("t_end", t_end)
    put("q", q)

    # Slider is 0 - 100 -> factor 0-1
    if el_slider_value is not None:
        out["el_size_factor"] = float(el_slider_value) / 100.0

    if param_steps is not None:
        out["param_steps"] = int(param_steps)

    # Checkboxes
    out["param_b"] = bool(vary_b)
    out["param_t"] = bool(vary_t)

    return out


#
# Streamlit state init
#
def init_state():
    default_params = Ps.ModelParams()
    default_dict = modelparams_to_dict(default_params)

    if "default_params_dict" not in st.session_state:
        st.session_state.default_params_dict = default_dict

    if "params_store" not in st.session_state:
        st.session_state.params_store = dict(default_dict)

    if "viewer_fig" not in st.session_state:
        st.session_state.viewer_fig = go.Figure()

    if "report" not in st.session_state:
        st.session_state.report = ""

    if "status" not in st.session_state:
        st.session_state.status = "Ready."

    if "current_filename" not in st.session_state:
        st.session_state.current_filename = None


def get_current_params_dict() -> dict:
    """Start from current params_store and overlay current UI values."""
    current_params = dict(st.session_state.default_params_dict)
    if isinstance(st.session_state.params_store, dict):
        current_params.update(st.session_state.params_store)

    # Pull widget values (set below) from session_state
    ui_patch = ui_values_to_params_dict(
        st.session_state.get("w_text"),
        st.session_state.get("h_text"),
        st.session_state.get("a_text"),
        st.session_state.get("b_text"),
        st.session_state.get("b_end_text"),
        st.session_state.get("E_text"),
        st.session_state.get("nu_text"),
        st.session_state.get("t_text"),
        st.session_state.get("t_end_text"),
        st.session_state.get("q_text"),
        st.session_state.get("El_size_Slider"),
        st.session_state.get("paramStep"),
        st.session_state.get("paramVaryB"),
        st.session_state.get("paramVaryT"),
    )

    # Merge patch into dict-like store
    current_params.update(ui_patch)
    return current_params


def build_modelparams_from_current_ui() -> Ps.ModelParams:
    mp = Ps.ModelParams()
    # start from stored params
    dict_to_modelparams(get_current_params_dict(), mp)
    return mp


def set_widgets_from_params_store():
    
    d = st.session_state.params_store
    if not isinstance(d, dict):
        d = dict(st.session_state.default_params_dict)

    el_factor = d.get("el_size_factor", 0.5)
    el_slider = int(float(el_factor) * 100)

    # Set defaults only if widget key not already in session_state,
    # otherwise Streamlit will complain about changing widget values after render.
    defaults = {
        "w_text": d.get("w"),
        "h_text": d.get("h"),
        "a_text": d.get("a"),
        "b_text": d.get("b"),
        "b_end_text": d.get("b_end"),
        "E_text": d.get("E"),
        "nu_text": d.get("nu"),
        "t_text": d.get("t"),
        "t_end_text": d.get("t_end"),
        "q_text": d.get("q"),
        "paramStep": d.get("param_steps", 10),
        "paramVaryB": bool(d.get("param_b", False)),
        "paramVaryT": bool(d.get("param_t", False)),
        "El_size_Slider": el_slider,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


#
# Actions
#
def action_new():
    st.session_state.params_store = dict(st.session_state.default_params_dict)
    st.session_state.viewer_fig = go.Figure()
    st.session_state.report = "New model created, reset to default parameters."
    st.session_state.status = "Ready."
    # Reset widget state so they reload from params_store next rerun
    for k in [
        "w_text","h_text","a_text","b_text","b_end_text","E_text","nu_text",
        "t_text","t_end_text","q_text","paramStep","paramVaryB","paramVaryT","El_size_Slider"
    ]:
        if k in st.session_state:
            del st.session_state[k]


def action_open(uploaded_file):
    if uploaded_file is None:
        st.session_state.status = "No file selected."
        return
    try:
        raw = uploaded_file.read()
        data = json.loads(raw.decode("utf-8"))

        mp = Ps.ModelParams()
        dict_to_modelparams(data, mp)

        st.session_state.params_store = modelparams_to_dict(mp)
        st.session_state.current_filename = uploaded_file.name
        st.session_state.viewer_fig = go.Figure()
        st.session_state.report = f"Loaded model from uploaded file: {uploaded_file.name}"
        st.session_state.status = "Loaded."

        # reset widgets to reflect loaded model
        for k in [
            "w_text","h_text","a_text","b_text","b_end_text","E_text","nu_text",
            "t_text","t_end_text","q_text","paramStep","paramVaryB","paramVaryT","El_size_Slider"
        ]:
            if k in st.session_state:
                del st.session_state[k]

    except Exception as ex:
        st.session_state.report = f"Error loading file: {ex}"
        st.session_state.status = "Load Failed"


def get_save_json_bytes():
    mp = build_modelparams_from_current_ui()
    data = modelparams_to_dict(mp)
    st.session_state.params_store = data
    json_str = json.dumps(data, indent=2)
    return json_str.encode("utf-8")


def action_execute(param_study: bool):
    try:
        mp = build_modelparams_from_current_ui()
        mr = Ps.ModelResult()
        solver = Ps.ModelSolver(mp, mr)

        if param_study:
            solver.execute_param_study()
            st.session_state.status = "Parameter study completed."
        else:
            solver.execute()
            st.session_state.status = "Execution completed."

        try:
            rep_obj = Ps.ModelReport(mp, mr)
            st.session_state.report = str(rep_obj)
        except Exception:
            st.session_state.report = "Execution completed. (No ModelReport available.)"

        st.session_state.params_store = modelparams_to_dict(mp)

    except Exception as ex:
        st.session_state.report = f"Error during execution: {ex}"
        st.session_state.status = "Execution Failed"


def action_visualize(which: str):
    try:
        mp = build_modelparams_from_current_ui()
        mr = Ps.ModelResult()
        solver = Ps.ModelSolver(mp, mr)
        solver.execute()

        vis = Ps.ModelVisualization(mp, mr)

        if which == "geometry":
            fig = vis.fig_geometry()
            st.session_state.status = "Geometry displayed."
        elif which == "mesh":
            fig = vis.fig_mesh()
            st.session_state.status = "Mesh displayed."
        elif which == "nodal":
            fig = vis.fig_displacements(magnfac=50.0)
            st.session_state.status = "Nodal displacements displayed."
        elif which == "element":
            fig = vis.fig_element_values()
            st.session_state.status = "Element stresses displayed."
        else:
            raise ValueError(f"Unknown visualization: {which}")

        st.session_state.viewer_fig = fig
        st.session_state.report = "Viewer updated."

        # keep params_store consistent with UI
        mp2 = build_modelparams_from_current_ui()
        st.session_state.params_store = modelparams_to_dict(mp2)

    except Exception as ex:
        st.session_state.report = f"Error during visualisation: {ex}"
        st.session_state.status = "Visualisation Failed"


#
# UI
#
def main():
    st.set_page_config(page_title="Plane Stress (Streamlit)", layout="wide")
    init_state()
    set_widgets_from_params_store()

    st.title("Plane Stress")

    # --- Menubar ---
    bar = st.columns([1, 2, 1, 1, 1, 1, 2])

    with bar[0]:
        if st.button("New"):
            action_new()
            st.rerun()

    with bar[1]:
        uploaded = st.file_uploader("Open", type=["json"], label_visibility="collapsed")
        # Act immediately after upload
        if uploaded is not None:
            action_open(uploaded)
            st.rerun()

    with bar[2]:
        # "Save" → download current model.json
        save_bytes = get_save_json_bytes()
        st.download_button(
            "Save",
            data=save_bytes,
            file_name="model.json",
            mime="application/json",
        )

    with bar[3]:
        # "Save as…" same behavior in this simple version
        saveas_bytes = get_save_json_bytes()
        st.download_button(
            "Save as…",
            data=saveas_bytes,
            file_name="model.json",
            mime="application/json",
        )

    with bar[4]:
        if st.button("Execute"):
            action_execute(param_study=False)
            st.rerun()

    with bar[5]:
        if st.button("Exit"):
            st.session_state.status = "Exit pressed (But it does nothing because this is a web application not a desktop application. Close the tab.)."

    st.divider()

    # --- Layout: Parameters / Viewer / Report ---
    st.subheader("Parameters")

        

    # w row + Geometry button
    c = st.columns([1, 2, 4, 2])
    c[0].markdown("**w [m]**")
    c[1].number_input(" ", key="w_text", format="%g")
    c[3].button("Geometry", on_click=action_visualize, args=("geometry",))
    
    # h row + Mesh button
    c = st.columns([1, 2, 4, 2])
    c[0].markdown("**h [m]**")
    c[1].number_input(" ", key="h_text", format="%g")
    c[3].button("Mesh", on_click=action_visualize, args=("mesh",))

    # a row + Displacements button
    c = st.columns([1, 2, 4, 2])
    c[0].markdown("**a [m]**")
    c[1].number_input(" ", key="a_text", format="%g")
    c[3].button("Displacements", on_click=action_visualize, args=("nodal",))

    # b row + Element values button + param study controls
    c = st.columns([1, 2, 1, 1.5, 1.5, 2])
    c[0].markdown("**b [m]**")
    c[1].number_input(" ", key="b_text", format="%g")
    c[2].markdown("**b, end**")
    c[3].number_input(" ", key="b_end_text", format="%g")
    c[4].checkbox("Vary", key="paramVaryB")
    c[5].button("Element Values", on_click=action_visualize, args=("element",))
    #st.button("Element Values", on_click=action_visualize, args=("element",))

    # E row
    c = st.columns([1, 2, 6])
    c[0].markdown("**E [Pa]**")
    c[1].number_input(" ", key="E_text", format="%g")
    
    # nu row
    c = st.columns([1, 2, 6])
    c[0].markdown("**v [-]**")
    c[1].number_input(" ", key="nu_text", format="%g")

    # t row + param study controls
    c = st.columns([1, 2, 1, 1.5, 3.5])
    c[0].markdown("**t [m]**")
    c[1].number_input(" ", key="t_text", format="%g")
    c[2].markdown("**t, end**")
    c[3].number_input(" ", key="t_end_text", format="%g")
    c[4].checkbox("Vary", key="paramVaryT")
    
    # q row + param study controls and button
    c = st.columns([1, 2, 1.5, 1, 1.5, 2])
    c[0].markdown("**q [N/m]**")
    c[1].number_input(" ", key="q_text", format="%g")
    c[3].markdown("**Param steps**")
    c[4].number_input(" ", key="paramStep", min_value=1, step=1)
    if c[5].button("Param Study"):
        action_execute(param_study=True)
        st.rerun()

    # El size slider row
    c = st.columns([1, 1, 5, 1.5])
    c[1].markdown("**El size**")
    c[2].slider(" ", key="El_size_Slider", min_value=0, max_value=100, step=1)
    c[3].markdown(f"**{st.session_state.El_size_Slider/100:.2f}**")


    st.subheader("Viewer")
    st.plotly_chart(st.session_state.viewer_fig, width="stretch", height=350)

    st.subheader("Report")
    st.text_area(
        " ",
        value=st.session_state.report,
        height=420,
        disabled=True,
    )

    st.caption(st.session_state.status)


if __name__ == "__main__":
    main()
