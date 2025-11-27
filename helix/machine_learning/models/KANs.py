import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch.nn import CrossEntropyLoss, Module
from tqdm import tqdm
from helix.options.enums import ProblemTypes
from copy import deepcopy
import sympy
import yaml


class KANMixin(KAN):

    def __init__(
        self,
        width: list[int],
        grid: int,
        k: int,
        seed: int,
        epochs: int,
        loss_fn: None | Module,
        lr: int,
        batch: int,
        problem_type: ProblemTypes,
        loading_model: bool,
    ):

        # NOTE: We do not initialise parent class KAN yet as we do not have the information of the number of features in or the number of nodes out.
        # To match scikit learn methods, we deduce that on the fit method and then create the NN architecture inplace.
        if loading_model:
            super().__init__(
                width=width,
                grid=grid,
                k=k,
                seed=seed,
                auto_save=False,
            )

        self.width = width
        self.grid = grid
        self.k = k
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch = batch
        self.problem_type = problem_type
        self.seed = seed
        self.loading_model = loading_model
        self._kan_initialized = False

    # this is the same fit function as in the original MultKAN class, but with small modifications to match with Helix's API
    def fit(
        self,
        # dataset,
        X,
        y,
        # opt="LBFGS",
        # steps=100,
        log=1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        update_grid=True,
        grid_update_num=10,
        # loss_fn=None,
        # lr=1.0,
        start_grid_update_step=-1,
        stop_grid_update_step=50,
        # batch=-1,
        metrics=None,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder="./video",
        singularity_avoiding=False,
        y_th=1000.0,
        reg_metric="edge_forward_spline_n",
        display_metrics=None,
    ):
        """
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        """

        # NOTE: this initialises the parent class KAN with the proper NN structure
        if not self._kan_initialized:
            in_feats = X.shape[1]
            out_nodes = (
                len(np.unique(y))
                if self.problem_type == ProblemTypes.Classification
                else 1
            )

            new_width = deepcopy(self.width)
            new_width.insert(0, in_feats)
            new_width.append(out_nodes)

            KAN.__init__(
                self,
                width=new_width,
                grid=self.grid,
                k=self.k,
                seed=self.seed,
                auto_save=False,
            )

            self._kan_initialized = True

        # take training arguments from the init method
        opt = "Adam"
        steps = self.epochs
        loss_fn = self.loss_fn
        lr = self.lr
        batch = self.batch

        if lamb > 0.0 and not self.save_act:
            print("setting lamb=0. If you want to set lamb > 0, set self.save_act=True")

        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc="description", ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        # only allow Adam for now
        # if opt == "Adam":

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # elif opt == "LBFGS":
        #     optimizer = LBFGS(
        #         self.get_params(),
        #         lr=lr,
        #         history_size=10,
        #         line_search_fn="strong_wolfe",
        #         tolerance_grad=1e-32,
        #         tolerance_change=1e-32,
        #         tolerance_ys=1e-32,
        #     )

        results = {}
        results["train_loss"] = []
        results["test_loss"] = []
        results["reg"] = []

        # create dataset dict to match the initial function logic.
        # No test input/label as this is not provided by helix until the training has finalised.
        dataset = {}

        X, y = self.sanitise_X(X), self.sanitise_y(y)

        dataset["train_input"] = X
        dataset["train_label"] = y
        dataset["test_input"] = X
        dataset["test_label"] = y

        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(
                dataset["train_input"][train_id],
                singularity_avoiding=singularity_avoiding,
                y_th=y_th,
            )
            train_loss = loss_fn(pred, dataset["train_label"][train_id])
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg_ = torch.tensor(0.0)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:

            if _ == steps - 1 and old_save_act:
                self.save_act = True

            if save_fig and _ % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True

            train_id = np.random.choice(
                dataset["train_input"].shape[0], batch_size, replace=False
            )
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if (
                _ % grid_update_freq == 0
                and _ < stop_grid_update_step
                and update_grid
                and _ >= start_grid_update_step
            ):
                self.update_grid(dataset["train_input"][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(
                    dataset["train_input"][train_id],
                    singularity_avoiding=singularity_avoiding,
                    y_th=y_th,
                )
                train_loss = loss_fn(pred, dataset["train_label"][train_id])
                if self.save_act:
                    if reg_metric == "edge_backward":
                        self.attribute()
                    if reg_metric == "node_backward":
                        self.node_attribute()
                    reg_ = self.get_reg(
                        reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                    )
                else:
                    reg_ = torch.tensor(0.0)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(
                self.forward(dataset["test_input"][test_id]),
                dataset["test_label"][test_id],
            )

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results["train_loss"].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results["reg"].append(reg_.cpu().detach().numpy())

            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description(
                        "| train_loss: %.2e | test_loss: %.2e | reg: %.2e | "
                        % (
                            torch.sqrt(train_loss).cpu().detach().numpy(),
                            torch.sqrt(test_loss).cpu().detach().numpy(),
                            reg_.cpu().detach().numpy(),
                        )
                    )
                else:
                    string = ""
                    data = ()
                    for metric in display_metrics:
                        string += f" {metric}: %.2e |"
                        try:
                            results[metric]
                        except:
                            raise Exception(f"{metric} not recognized")
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and _ % save_fig_freq == 0:
                self.plot(
                    folder=img_folder,
                    in_vars=in_vars,
                    out_vars=out_vars,
                    title="Step {}".format(_),
                    beta=beta,
                )
                plt.savefig(
                    img_folder + "/" + str(_) + ".jpg", bbox_inches="tight", dpi=200
                )
                plt.close()
                self.save_act = save_act

        self.log_history("fit")
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results

    def sanitise_X(self, X: pd.DataFrame | np.ndarray):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = torch.tensor(X, dtype=torch.float32)

        return X

    def sanitise_y(self, y: pd.Series | np.ndarray):
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        return y

    def saveckpt(self, path="model"):
        """
        save the current model to files (configuration file and state file)

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        # There will be three files appearing in the current folder: mark_cache_data, mark_config.yml, mark_state
        """

        model = self

        dic = dict(
            width=model.width,
            grid=model.grid,
            k=model.k,
            epochs=model.epochs,
            loss_fn=model.loss_fn,
            lr=model.lr,
            batch=model.batch,
            seed=model.seed,
            problem_type=model.problem_type.value,
            classes=(
                model.classes_
                if model.problem_type.value == ProblemTypes.Classification
                else None
            ),
            _kan_initialized=model._kan_initialized,
            auto_save=model.auto_save,
            ckpt_path=model.ckpt_path,
            round=model.round,
            device=str(model.device),
        )

        if dic["device"].isdigit():
            dic["device"] = int(model.device)

        for i in range(model.depth):
            dic[f"symbolic.funs_name.{i}"] = model.symbolic_fun[i].funs_name

        with open(f"{path}_config.yml", "w") as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        torch.save(model.state_dict(), f"{path}_state")
        torch.save(model.cache_data, f"{path}_cache_data")

    @staticmethod
    def loadckpt(path="model"):
        """
        load checkpoint from path

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            MultKAN

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        >>> KAN.loadckpt('./mark')
        """
        with open(f"{path}_config.yml", "r") as stream:
            config = yaml.safe_load(stream)

        state = torch.load(f"{path}_state")

        problem_type = config["problem_type"]

        if problem_type == ProblemTypes.Regression:
            kan_model = KANRegressor(
                width=config["width"],
                grid=config["grid"],
                k=config["k"],
                epochs=config["epochs"],
                lr=config["lr"],
                batch=config["batch"],
                seed=config["seed"],
                loading_model=True,
            )
        elif problem_type == ProblemTypes.Classification:
            kan_model = KANClassifier(
                width=config["width"],
                grid=config["grid"],
                k=config["k"],
                epochs=config["epochs"],
                lr=config["lr"],
                batch=config["batch"],
                seed=config["seed"],
                loading_model=True,
                classes=config["classes"],
            )

        model_load = kan_model
        # model_load = kan_model(
        #     width=config["width"],
        #     grid=config["grid"],
        #     k=config["k"],
        #     epochs=config["epochs"],
        #     lr=config["lr"],
        #     batch=config["batch"],
        #     seed=config["seed"],
        #     loading_model=True,
        # mult_arity=config["mult_arity"],
        # base_fun=config["base_fun_name"],
        # symbolic_enabled=config["symbolic_enabled"],
        # affine_trainable=config["affine_trainable"],
        # grid_eps=config["grid_eps"],
        # grid_range=config["grid_range"],
        # sp_trainable=config["sp_trainable"],
        # sb_trainable=config["sb_trainable"],
        # state_id=config["state_id"],
        # auto_save=config["auto_save"],
        # first_init=False,
        # ckpt_path=config["ckpt_path"],
        # round=config["round"] + 1,
        # device=config["device"],
        # )

        model_load.load_state_dict(state)
        model_load.cache_data = torch.load(f"{path}_cache_data")

        depth = len(model_load.width) - 1
        for l in range(depth):
            out_dim = model_load.symbolic_fun[l].out_dim
            in_dim = model_load.symbolic_fun[l].in_dim
            funs_name = config[f"symbolic.funs_name.{l}"]
            for j in range(out_dim):
                for i in range(in_dim):
                    fun_name = funs_name[j][i]
                    model_load.symbolic_fun[l].funs_name[j][i] = fun_name
                    model_load.symbolic_fun[l].funs[j][i] = SYMBOLIC_LIB[fun_name][0]
                    model_load.symbolic_fun[l].funs_sympy[j][i] = SYMBOLIC_LIB[
                        fun_name
                    ][1]
                    model_load.symbolic_fun[l].funs_avoid_singularity[j][i] = (
                        SYMBOLIC_LIB[fun_name][3]
                    )
        return model_load


class KANClassifier(ClassifierMixin, BaseEstimator, KANMixin):

    def __init__(
        self,
        width: list = None,
        grid: int = 5,
        k: int = 3,
        epochs: int = 100,
        lr: float = 1,
        batch: int = -1,
        seed: int = 42,
        loading_model=False,
        classes=None,
    ):

        # TODO: correct to use appropriate loss when it is fixed in the future
        super().__init__(
            width=width,
            grid=grid,
            k=k,
            seed=seed,
            epochs=epochs,
            loss_fn=None,
            # loss_fn=CrossEntropyLoss(),
            lr=lr,
            batch=batch,
            problem_type=ProblemTypes.Classification,
            loading_model=loading_model,
        )
        self.classes = None
        self.classes_ = None
        if loading_model:
            self.classes_ = classes

    def fit(self, X, y):

        super().fit(X, y)
        classes = np.unique(y).tolist()
        self.classes_ = classes
        self.classes = classes

        return self

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = self.sanitise_X(X)
        y_pred_prob = self.forward(X).detach().numpy()
        return y_pred_prob


class KANRegressor(RegressorMixin, BaseEstimator, KANMixin):

    def __init__(
        self,
        width: list = None,
        grid: int = 5,
        k: int = 3,
        epochs: int = 100,
        lr: float = 1,
        batch: int = -1,
        seed: int = 42,
        loading_model=False,
    ):

        super().__init__(
            width=width,
            grid=grid,
            k=k,
            seed=seed,
            epochs=epochs,
            loss_fn=None,
            lr=lr,
            batch=batch,
            problem_type=ProblemTypes.Regression,
            loading_model=loading_model,
        )

    def predict(self, X):

        X = self.sanitise_X(X)
        y_pred = self.forward(X).detach().numpy().ravel()

        return y_pred


f_inv = lambda x, y_th: (
    (x_th := 1 / y_th),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x) * (torch.abs(x) >= x_th),
)
f_inv2 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 2)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**2) * (torch.abs(x) >= x_th),
)
f_inv3 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 3)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**3) * (torch.abs(x) >= x_th),
)
f_inv4 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 4)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**4) * (torch.abs(x) >= x_th),
)
f_inv5 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 5)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**5) * (torch.abs(x) >= x_th),
)
f_sqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    x_th / y_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.sqrt(torch.abs(x)) * torch.sign(x))
    * (torch.abs(x) >= x_th),
)
f_power1d5 = lambda x, y_th: torch.abs(x) ** 1.5
f_invsqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_log = lambda x, y_th: (
    (x_th := torch.e ** (-y_th)),
    -y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_tan = lambda x, y_th: (
    (clip := x % torch.pi),
    (delta := torch.pi / 2 - torch.arctan(y_th)),
    -y_th / delta * (clip - torch.pi / 2) * (torch.abs(clip - torch.pi / 2) < delta)
    + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi / 2) >= delta),
)
f_arctanh = lambda x, y_th: (
    (delta := 1 - torch.tanh(y_th) + 1e-4),
    y_th * torch.sign(x) * (torch.abs(x) > 1 - delta)
    + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta),
)
f_arcsin = lambda x, y_th: (
    (),
    torch.pi / 2 * torch.sign(x) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1),
)
f_arccos = lambda x, y_th: (
    (),
    torch.pi / 2 * (1 - torch.sign(x)) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1),
)
f_exp = lambda x, y_th: (
    (x_th := torch.log(y_th)),
    y_th * (x > x_th) + torch.exp(x) * (x <= x_th),
)

SYMBOLIC_LIB = {
    "x": (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
    "x^2": (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
    "x^3": (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
    "x^4": (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
    "x^5": (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
    "1/x": (lambda x: 1 / x, lambda x: 1 / x, 2, f_inv),
    "1/x^2": (lambda x: 1 / x**2, lambda x: 1 / x**2, 2, f_inv2),
    "1/x^3": (lambda x: 1 / x**3, lambda x: 1 / x**3, 3, f_inv3),
    "1/x^4": (lambda x: 1 / x**4, lambda x: 1 / x**4, 4, f_inv4),
    "1/x^5": (lambda x: 1 / x**5, lambda x: 1 / x**5, 5, f_inv5),
    "sqrt": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    "x^0.5": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    "x^1.5": (
        lambda x: torch.sqrt(x) ** 3,
        lambda x: sympy.sqrt(x) ** 3,
        4,
        f_power1d5,
    ),
    "1/sqrt(x)": (
        lambda x: 1 / torch.sqrt(x),
        lambda x: 1 / sympy.sqrt(x),
        2,
        f_invsqrt,
    ),
    "1/x^0.5": (lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2, f_invsqrt),
    "exp": (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
    "log": (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
    "abs": (
        lambda x: torch.abs(x),
        lambda x: sympy.Abs(x),
        3,
        lambda x, y_th: ((), torch.abs(x)),
    ),
    "sin": (
        lambda x: torch.sin(x),
        lambda x: sympy.sin(x),
        2,
        lambda x, y_th: ((), torch.sin(x)),
    ),
    "cos": (
        lambda x: torch.cos(x),
        lambda x: sympy.cos(x),
        2,
        lambda x, y_th: ((), torch.cos(x)),
    ),
    "tan": (lambda x: torch.tan(x), lambda x: sympy.tan(x), 3, f_tan),
    "tanh": (
        lambda x: torch.tanh(x),
        lambda x: sympy.tanh(x),
        3,
        lambda x, y_th: ((), torch.tanh(x)),
    ),
    "sgn": (
        lambda x: torch.sign(x),
        lambda x: sympy.sign(x),
        3,
        lambda x, y_th: ((), torch.sign(x)),
    ),
    "arcsin": (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
    "arccos": (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
    "arctan": (
        lambda x: torch.arctan(x),
        lambda x: sympy.atan(x),
        4,
        lambda x, y_th: ((), torch.arctan(x)),
    ),
    "arctanh": (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
    "0": (lambda x: x * 0, lambda x: x * 0, 0, lambda x, y_th: ((), x * 0)),
    "gaussian": (
        lambda x: torch.exp(-(x**2)),
        lambda x: sympy.exp(-(x**2)),
        3,
        lambda x, y_th: ((), torch.exp(-(x**2))),
    ),
    #'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5),
    #'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid'), 4),
    #'relu': (lambda x: torch.relu(x), relu),
}
