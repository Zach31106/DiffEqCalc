import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, Eq, solve, lambdify
from sympy.parsing.sympy_parser import parse_expr
from scipy.integrate import odeint

def step_function(x):
    """Unit-step function used by the calculator."""
    return np.where(x >= 0, 1, 0)


def impulse_function(a):
    """Gaussian approximation to the Dirac-delta function."""
    return (1 / (0.005 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (a / 0.005) ** 2
    )


def slope_field(f, x_range, y_range, step=0.2):
    x, y = np.meshgrid(
        np.arange(x_range[0], x_range[1], step),
        np.arange(y_range[0], y_range[1], step),
    )
    u = np.ones_like(x)
    v = f(x, y)
    norm = np.sqrt(u**2 + v**2)
    norm = np.where(norm == 0, 1, norm)
    return x, y, u / norm, v / norm


def plot_slope_field(f, x_range, y_range, ax):
    x, y, u, v = slope_field(f, x_range, y_range)
    ax.quiver(x, y, u, v, color="blue")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True)


def draw_chart(x, y, xl, xr, slope=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="black")

    if slope is not None:
        plot_slope_field(slope, (xl, xr), ax.get_ylim(), ax)

    ax.set_xlim((xl, xr))
    ax.grid(axis="both", color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=1)


    fig.tight_layout()
    return fig


def graph_input(display_str, parser_str, xl_str, xr_str, ivp1, ivp2):
    """
    Convert the calculator entry into either:
      - a second-order ODE containing y2
      - a first-order ODE containing y1
      - an explicit y(x) equation containing y0
      - a constant expression
    """
    xl = float((xl_str or "").strip() or 0)
    xr = float((xr_str or "").strip() or 1)

    if xr == xl:
        raise ValueError("The x-range must have different left and right bounds.")

    input_string = parser_str.strip()
    if not input_string:
        raise ValueError("Enter an equation before pressing Graph.")

    if "=" in input_string:
        lhs_text, rhs_text = input_string.split("=", 1)
        lhs = parse_expr(lhs_text.replace("np.", ""))
        rhs = parse_expr(rhs_text.replace("np.", ""))
    else:
        lhs = None
        rhs = None

    if "y2" in input_string:
        if "=" not in input_string:
            raise SyntaxError("Second-order equations must contain '='.")

        x, y2, y1, y0 = symbols("x y2 y1 y0")
        differential_equation = Eq(lhs, rhs)
        solved_y2 = solve(differential_equation, y2)

        if not solved_y2:
            raise ValueError("Could not solve the equation for y''.")

        y2_function = lambdify(
            (x, y0, y1),
            solved_y2[0],
            modules=[
                "numpy",
                {"step_function": step_function},
                {"impulse_function": impulse_function},
            ],
        )

        time = np.linspace(xl, xr, 1001)
        init_conditions = [ivp1, ivp2]

        def system_of_odes(values, t):
            y_value, y_prime_value = values
            return y_prime_value, y2_function(t, y_value, y_prime_value)

        solution = odeint(system_of_odes, init_conditions, time)

        return draw_chart(
            time,
            solution[:, 0],
            xl,
            xr,
            None,
        )

    if "y1" in input_string:
        if "=" not in input_string:
            raise SyntaxError("First-order differential equations must contain '='.")

        x, y1, y0 = symbols("x y1 y0")
        differential_equation = Eq(lhs, rhs)
        solved_y1 = solve(differential_equation, y1)

        if not solved_y1:
            raise ValueError("Could not solve the equation for y'.")

        y1_function = lambdify(
            (x, y0),
            solved_y1[0],
            modules=[
                "numpy",
                {"step_function": step_function},
                {"impulse_function": impulse_function},
            ],
        )

        time = np.linspace(xl, xr, 1001)

        def system_of_odes(y_value, t):
            return y1_function(t, y_value)

        solution = odeint(system_of_odes, ivp1, time)

        return draw_chart(
            time,
            solution[:],
            xl,
            xr,
            y1_function,
        )

    if "y0" in input_string:
        if "=" not in input_string:
            raise SyntaxError("Explicit equations must contain '='.")

        x, y0 = symbols("x y0")
        equation = Eq(lhs, rhs)
        solved_y0 = solve(equation, y0)

        if not solved_y0:
            raise ValueError("Could not solve the equation for y.")

        y_func = lambdify(
            x,
            solved_y0[0],
            modules=[
                "numpy",
                {"step_function": step_function},
                {"impulse_function": impulse_function},
            ],
        )

        x_vals = np.linspace(xl, xr, 1001)
        y_vals = y_func(x_vals)

        return draw_chart(
            x_vals,
            y_vals,
            xl,
            xr,
            None,
        )

    if "=" in input_string:
        raise SyntaxError(
            "The equation must contain x, y, y', or y'' in a recognizable form."
        )

    x_vals = np.linspace(xl, xr, 1001)

    allowed_names = {
        "np": np,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "log": np.log,
        "exp": np.exp,
        "pi": np.pi,
        "e": np.e,
        "abs": np.abs,
    }

    constant_value = eval(
        input_string.replace("^", "**"),
        {"__builtins__": {}},
        allowed_names,
    )
    y_vals = np.full_like(x_vals, float(constant_value), dtype=float)

    return draw_chart(x_vals, y_vals, xl, xr, None)

st.set_page_config(
    page_title="Differential Equation Visualizer",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: 1600px;
    }

    div[data-testid="column"] > div {
        margin-bottom: 1.4rem !important;
    }

    .stButton > button {
        padding: 0.15rem 0.3rem !important;
        margin: 0.1rem !important;
    }

    div[data-testid="stTextInput"] input {
        font-family: monospace;
    }

    .display-box {
        border: 1px solid #bbb;
        border-radius: 4px;
        background: white;
        color: black;
        padding: 0.55rem 0.7rem;
        min-height: 2.5rem;
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

defaults = {
    "equationText": "",
    "calculationText": "",
    "input_stack": [],
    "spanXL": "0",
    "spanXR": "10",
    "ivp1_input": "",
    "ivp2_input": "",
    "IVP1": 0,
    "IVP2": 0,
    "last_fig": None,
    "last_display_equation": "",
    "last_parser_equation": "",
    "last_plotted_spanXL": None,
    "last_plotted_spanXR": None,
    "last_plotted_IVP1": None,
    "last_plotted_IVP2": None,
    "error_message": None,
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


def rerender():
    """Small helper for actions that should immediately refresh the UI."""
    st.rerun()


def input_entry(display_str, parser_str):
    st.session_state.equationText += str(display_str)
    st.session_state.calculationText += str(parser_str)
    st.session_state.input_stack.append((str(display_str), str(parser_str)))


def delete_last_input():
    if not st.session_state.input_stack:
        return

    display_value, parser_value = st.session_state.input_stack.pop()

    st.session_state.equationText = st.session_state.equationText.rsplit(
        display_value, 1
    )[0]
    st.session_state.calculationText = st.session_state.calculationText.rsplit(
        parser_value, 1
    )[0]


def set_value(which, text):
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        label = "y(x1)" if which == "IVP1" else "y'(x1)"
        st.session_state.error_message = f"{label} must be an integer."
        return

    if which == "IVP1":
        st.session_state.IVP1 = value
    elif which == "IVP2":
        st.session_state.IVP2 = value

    st.session_state.error_message = None


def reset_calculator():
    """Reset both calculator state and the editable IVP widgets."""
    st.session_state.equationText = ""
    st.session_state.calculationText = ""
    st.session_state.input_stack = []
    st.session_state.last_fig = None
    st.session_state.last_display_equation = ""
    st.session_state.last_parser_equation = ""
    st.session_state.last_plotted_spanXL = None
    st.session_state.last_plotted_spanXR = None
    st.session_state.last_plotted_IVP1 = None
    st.session_state.last_plotted_IVP2 = None
    st.session_state.error_message = None
    st.session_state.IVP1 = 0
    st.session_state.IVP2 = 0

    st.session_state.pop("ivp1_input", None)
    st.session_state.pop("ivp2_input", None)


def format_equation_latex(display_text):
    text = (display_text or "").strip()
    if not text:
        return ""

    replacements = [
        ("y''", r"y''"),
        ("arcsin(", r"\arcsin("),
        ("arccos(", r"\arccos("),
        ("arctan(", r"\arctan("),
        ("sin(", r"\sin("),
        ("cos(", r"\cos("),
        ("tan(", r"\tan("),
        ("ln(", r"\ln("),
        ("abs(", r"\operatorname{abs}("),
        ("u(", r"u("),
        ("δ(", r"\delta("),
        ("π", r"\pi"),
        ("√(", r"\sqrt{"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    text = text.replace("^(", "^{")
    if "^{" in text:
        text = text.replace(")", "}", 1)

    if "\\sqrt{" in text:
        open_idx = text.find("\\sqrt{")
        if open_idx >= 0:
            close_idx = text.find(")", open_idx)
            if close_idx >= 0:
                text = text[:close_idx] + "}" + text[close_idx + 1:]

    return text


def replot_saved_equation_if_inputs_changed():
    if not st.session_state.last_parser_equation:
        return

    try:
        xl = float((st.session_state.spanXL or "").strip() or 0)
        xr = float((st.session_state.spanXR or "").strip() or 1)
        ivp1 = int(str(st.session_state.IVP1))
        ivp2 = int(str(st.session_state.IVP2))

        current = (xl, xr, ivp1, ivp2)
        previous = (
            st.session_state.last_plotted_spanXL,
            st.session_state.last_plotted_spanXR,
            st.session_state.last_plotted_IVP1,
            st.session_state.last_plotted_IVP2,
        )
        if current == previous:
            return

        st.session_state.last_fig = graph_input(
            st.session_state.last_display_equation,
            st.session_state.last_parser_equation,
            str(xl),
            str(xr),
            ivp1,
            ivp2,
        )
        st.session_state.last_plotted_spanXL = xl
        st.session_state.last_plotted_spanXR = xr
        st.session_state.last_plotted_IVP1 = ivp1
        st.session_state.last_plotted_IVP2 = ivp2
        st.session_state.error_message = None
    except Exception as exc:
        st.session_state.error_message = f"Error updating graph: {exc}"


def initial_figure():
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(-10, 10, 1000)
    y = np.full_like(x, np.nan)
    ax.plot(x, y, color="black")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.grid(axis="both", color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig

left, right = st.columns([1, 1])

with left:
    with st.container(border=True):
        st.markdown("**X range**")
        span_left, span_operator, span_right = st.columns([1, 0.5, 1])

        with span_left:
            st.session_state.spanXL = st.text_input(
                "Left bound",
                value=st.session_state.spanXL,
                key="spanXL_input",
            )

        with span_operator:
            st.markdown(
                "<div style='text-align:center; padding-top:1.95rem; font-size:1.05rem;'>"
                "≤ x ≤"
                "</div>",
                unsafe_allow_html=True,
            )

        with span_right:
            st.session_state.spanXR = st.text_input(
                "Right bound",
                value=st.session_state.spanXR,
                key="spanXR_input",
            )

    st.markdown("**Equation**")
    st.markdown(
        f"<div class='display-box'>{st.session_state.equationText or '&nbsp;'}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.last_display_equation:
        st.markdown("**Graphed equation**")
        st.latex(format_equation_latex(st.session_state.last_display_equation))

    iv1, iv2 = st.columns(2)

    with iv1:
        ivp1_text = st.text_input("y(x1)", key="ivp1_input")

    with iv2:
        ivp2_text = st.text_input("y'(x1)", key="ivp2_input")

    try:
        st.session_state.IVP1 = int(ivp1_text.strip() or 0)
        st.session_state.IVP2 = int(ivp2_text.strip() or 0)
        st.session_state.error_message = None
    except ValueError:
        st.session_state.error_message = "Initial conditions must be integers."

replot_saved_equation_if_inputs_changed()

with right:
    if st.session_state.last_fig is not None:
        st.pyplot(st.session_state.last_fig, clear_figure=False)
    else:
        st.pyplot(initial_figure(), clear_figure=False)

mapping = {
    "x": "x",
    "y": "y0",
    "y'": "y1",
    "y''": "y2",
    "7": "7",
    "8": "8",
    "9": "9",
    "/": "/",
    "sin()": "np.sin(",
    "cos()": "np.cos(",
    "tan()": "np.tan(",
    "^2": "**2",
    "^a": "**(",
    "(": "(",
    ")": ")",
    "4": "4",
    "5": "5",
    "6": "6",
    "*": "*",
    "arcsin()": "np.arcsin(",
    "arccos()": "np.arccos(",
    "arctan()": "np.arctan(",
    "||": "abs(",
    "√": "np.sqrt(",
    "π": "np.pi",
    "e": "np.exp(1)",
    "1": "1",
    "2": "2",
    "3": "3",
    "-": "-",
    "ln()": "np.log(",
    "u(x-a)": "step_function(",
    "δ(x-a)": "impulse_function(",
    "0": "0",
    ".": ".",
    "=": "=",
    "+": "+",
}

display_mapping = {
    "^a": "^(",
    "sin()": "sin(",
    "cos()": "cos(",
    "tan()": "tan(",
    "arcsin()": "arcsin(",
    "arccos()": "arccos(",
    "arctan()": "arctan(",
    "ln()": "ln(",
    "u(x-a)": "u(",
    "δ(x-a)": "δ(",
    "√": "√(",
    "||": "abs(",
}


def make_input_button(label, row_number):
    if st.button(label, key=f"input_{row_number}_{label}", use_container_width=True):
        input_entry(display_mapping.get(label, label), mapping[label])
        st.rerun()

def row(buttons, row_number):
    cols = st.columns(11)
    for i, button_label in enumerate(buttons):
        with cols[i]:
            if not button_label:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                continue
            make_input_button(button_label, f"{row_number}_{i}")


row(
    ["x", "y", "y'", "y''", "7", "8", "9", "/", "sin()", "cos()", "tan()"],
    1,
)

row(
    ["^2", "^a", "(", ")", "4", "5", "6", "*", "arcsin()", "arccos()", "arctan()"],
    2,
)

row(
    ["||", "√", "π", "e", "1", "2", "3", "-", "ln()", "u(x-a)", "δ(x-a)"],
    3,
)

action_cols = st.columns(11)

with action_cols[4]:
    if st.button("0", key="input_4_4", use_container_width=True):
        input_entry("0", "0")
        st.rerun()

with action_cols[5]:
    if st.button(".", key="input_4_5", use_container_width=True):
        input_entry(".", ".")
        st.rerun()

with action_cols[6]:
    if st.button("=", key="input_4_6", use_container_width=True):
        input_entry("=", "=")
        st.rerun()

with action_cols[7]:
    if st.button("+", key="input_4_7", use_container_width=True):
        input_entry("+", "+")
        st.rerun()

with action_cols[8]:
    if st.button("Graph", key="graph_button", use_container_width=True):
        try:
            fig = graph_input(
                st.session_state.equationText,
                st.session_state.calculationText,
                st.session_state.spanXL,
                st.session_state.spanXR,
                st.session_state.IVP1,
                st.session_state.IVP2,
            )
            st.session_state.last_fig = fig
            st.session_state.last_display_equation = st.session_state.equationText
            st.session_state.last_parser_equation = st.session_state.calculationText

            st.session_state.last_plotted_spanXL = float((st.session_state.spanXL or "").strip() or 0)
            st.session_state.last_plotted_spanXR = float((st.session_state.spanXR or "").strip() or 1)
            st.session_state.last_plotted_IVP1 = st.session_state.IVP1
            st.session_state.last_plotted_IVP2 = st.session_state.IVP2

            st.session_state.equationText = ""
            st.session_state.calculationText = ""
            st.session_state.input_stack = []
            st.session_state.error_message = None

        except Exception as exc:
            st.session_state.error_message = f"Error: {exc}"
        st.rerun()

with action_cols[9]:
    if st.button("Delete", key="delete_button", use_container_width=True):
        delete_last_input()
        st.rerun()

with action_cols[10]:
    if st.button("Reset", key="reset_button", use_container_width=True):
        reset_calculator()
        st.rerun()

if st.session_state.error_message:
    st.error(st.session_state.error_message)
