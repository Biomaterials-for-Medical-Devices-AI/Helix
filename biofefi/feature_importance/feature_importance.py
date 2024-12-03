from biofefi.feature_importance.interpreter import Interpreter
from biofefi.options.execution import ExecutionOptions
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.plotting import PlottingOptions


def run(
    fi_opt: FeatureImportanceOptions,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    data,
    models,
    logger,
):

    # Interpret the model results
    interpreter = Interpreter(
        fi_opt=fi_opt, exec_opt=exec_opt, plot_opt=plot_opt, logger=logger
    )
    # TODO: Add indices to the dataframe results-> global + ensemble
    # TODO: Resolve majority vote issue
    gloabl_importance_results, local_importance_results, ensemble_results = (
        interpreter.interpret(models, data)
    )

    return gloabl_importance_results, local_importance_results, ensemble_results
