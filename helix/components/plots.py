import json
from pathlib import Path

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


@st.experimental_fragment
def plot_box(plot_dir: Path, box_title: str):
    """Display the plots in the given directory in the UI.

    Args:
        plot_dir (Path): The directory containing the plots.
        box_title (str): The title of the plot box.
    """
    if plot_dir.exists():
        plots = sorted(plot_dir.iterdir(), key=lambda x: x.stat().st_ctime)
        with st.expander(box_title, expanded=len(plots) > 0):
            # Group plots by model and type
            plot_groups = {}
            for p in plots:
                if not p.name.endswith(".png"):
                    continue

                # Extract model name and plot type from filename
                parts = p.stem.split("-")
                if len(parts) >= 2:
                    model_name = parts[0]
                    if model_name not in plot_groups:
                        plot_groups[model_name] = {
                            "Train": [],
                            "Test": [],
                            "Coefficients": [],
                            "Other": [],
                        }

                    if "Train" in parts:
                        plot_groups[model_name]["Train"].append(p)
                    elif "Test" in parts:
                        plot_groups[model_name]["Test"].append(p)
                    elif "coefficients" in parts:
                        plot_groups[model_name]["Coefficients"].append(p)
                    else:
                        plot_groups[model_name]["Other"].append(p)

            # Display plots by group
            for model_name, group in plot_groups.items():
                # Capitalise first letter of each word in model name
                formatted_model_name = " ".join(
                    word.capitalize() for word in model_name.split()
                )
                st.markdown(f"### {formatted_model_name}")

                # Show Train/Test plots side by side
                if group["Train"] or group["Test"]:
                    cols = st.columns(2)
                    for train_plot in group["Train"]:
                        cols[0].image(str(train_plot))
                    for test_plot in group["Test"]:
                        cols[1].image(str(test_plot))

                # Show coefficient plots
                for coef_plot in group["Coefficients"]:
                    st.markdown("#### Model Coefficients")
                    st.image(str(coef_plot))

                # Show other plots
                for other_plot in group["Other"]:
                    st.image(str(other_plot))

                st.markdown("---")


@st.experimental_fragment
def display_metrics_table(metrics_path: Path):
    """
    Display a metrics summary table in a Streamlit app.

    Args:
        metrics_path (Path): The path to the metrics file.
    """
    # Check if metrics file exists
    if not metrics_path.exists():
        st.info("No metrics available yet. Train some models first.")
        return

    # Load the metrics from the file
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)

    # Prepare data for the table
    rows = []
    for algorithm, results in metrics_dict.items():
        for dataset, metrics in results.items():
            for metric, values in metrics.items():
                row = {
                    "Model": algorithm,
                    "Set": dataset.capitalize(),
                    "Metric": metric,
                    "Mean ± Std": f"{values['mean']:.3f} ± {values['std']:.3f}",
                }
                rows.append(row)

    # Create a DataFrame
    df = pd.DataFrame(rows)

    # Pivot the DataFrame for a cleaner table
    table = df.pivot(
        index=["Model", "Set"], columns="Metric", values="Mean ± Std"
    ).reset_index()
    table = table.set_index("Model")
    table.sort_values(["Model", "Set"], ascending=[True, True], inplace=True)

    # Display the table in Streamlit
    st.write("### Metrics Summary")
    st.write(
        "Metrics are the mean (± standard deviation) of all bootstraps (if using the Holdout"
        " data split) or cross-validation folds (if using K-fold data split or"
        " automatic hyper-parameter search)."
    )
    # TODO: This can be moved to a separate function
    # Build Grid Options
    table = table.reset_index()
    gb = GridOptionsBuilder.from_dataframe(table)

    # Apply Global Font Styling to All Columns
    global_style = {
        "fontSize": "14px",
        "fontFamily": "Arial, sans-serif",
        "color": "black",
        "textAlign": "center",
    }

    gb.configure_default_column(
        editable=False, resizable=True, flex=2, cellStyle=global_style, wrapText=False
    )
    gb.configure_grid_options(domLayout="autoHeight")
    gb.configure_first_column_as_index()
    grid_options = gb.build()

    # Display Full-Width Ag-Grid Table with Styling
    AgGrid(
        table,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        highlight_odd_rows=True,
        key="metrics_table",
    )


@st.experimental_fragment
def display_predictions(predictions_df: pd.DataFrame):
    """
    Display the predictions in the UI.

    Args:
        predictions_path (Path): The path to the predictions file.
    """
    st.write("### Predictions")
    st.write(predictions_df)
