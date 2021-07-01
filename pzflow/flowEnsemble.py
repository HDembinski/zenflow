import jax.numpy as np
from jax.experimental.optimizers import Optimizer
import pandas as pd
from pzflow import Flow
from typing import Any, Sequence, Tuple, Callable
from pzflow.bijectors import InitFunction, Bijector_Info
import dill as pickle


class FlowEnsemble:
    """An ensemble of normalizing flows.

    Attributes
    ----------
    data_columns : tuple
        List of DataFrame columns that the flows expect/produce.
    conditional_columns : tuple
        List of DataFrame columns on which the flows are conditioned.
    info : Any
        Object containing any kind of info included with the ensemble.
        Often Reverse the data the flows are trained on.
    latent
        The latent distribution of the normalizing flows.
        Has it's own sample and log_prob methods.
    """

    def __init__(
        self,
        data_columns: Sequence[str] = None,
        bijector: Tuple[InitFunction, Bijector_Info] = None,
        conditional_columns: Sequence[str] = None,
        latent=None,
        N: int = 1,
        info: Any = None,
        file: str = None,
    ):
        """Instantiate an ensemble of normalizing flows.

        Note that while all of the init parameters are technically optional,
        you must provide either data_columns and bijector OR file.
        In addition, if a file is provided, all other parameters must be None.

        Parameters
        ----------
        data_columns : Sequence[str], optional
            Tuple, list, or other container of column names.
            These are the columns the flows expect/produce in DataFrames.
        bijector : Bijector Call, optional
            A Bijector call that consists of the bijector InitFunction that
            initializes the bijector and the tuple of Bijector Info.
            Can be the output of any Bijector, e.g. Reverse(), Chain(...), etc.
        conditional_columns : Sequence[str], optional
            Names of columns on which to condition the normalizing flows.
        latent : distribution, optional
            The latent distribution for the normalizing flows. Can be any of
            the distributions from pzflow.distributions. If not provided,
            a normal distribution is used with the number of dimensions
            inferred.
        N : int, default=1
            The number of flows in the ensemble.
        info : Any, optional
            An object to attach to the info attribute.
        file : str, optional
            Path to file from which to load a pretrained flow ensemble.
            If a file is provided, all other parameters must be None.
        """

        # validate parameters
        if data_columns is None and bijector is None and file is None:
            raise ValueError("You must provide data_columns and bijector OR file.")
        if data_columns is not None and bijector is None:
            raise ValueError("Please also provide a bijector.")
        if data_columns is None and bijector is not None:
            raise ValueError("Please also provide data_columns.")
        if file is not None and any(
            (
                data_columns is not None,
                bijector is not None,
                conditional_columns is not None,
                latent is not None,
                info is not None,
            )
        ):
            raise ValueError(
                "If providing a file, please do not provide any other parameters."
            )

        # if file is provided, load everything from the file
        if file is not None:

            # load the file
            with open(file, "rb") as handle:
                save_dict = pickle.load(handle)

            # make sure the saved file is for this class
            c = save_dict.pop("class")
            if c != self.__class__.__name__:
                raise TypeError(
                    f"This save file isn't a {self.__class__.__name__}. It is a {c}."
                )

            # load the ensemble from the dictionary
            self._ensemble = {
                name: Flow(_dictionary=flow_dict)
                for name, flow_dict in save_dict.items()
            }
        # otherwise create a new ensemble from the provided parameters
        else:
            self._ensemble = {
                f"Flow {i}": Flow(
                    data_columns=data_columns,
                    bijector=bijector,
                    conditional_columns=conditional_columns,
                    latent=latent,
                    seed=i,
                )
                for i in range(N)
            }

    def log_prob(
        self,
        inputs: pd.DataFrame,
        nsamples: int = None,
        seed: int = None,
        returnEnsemble: bool = False,
    ) -> np.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Input data for which log probability density is calculated.
            Every column in self.data_columns must be present.
            If self.conditional_columns is not None, those must be present
            as well. If other columns are present, they are ignored.
        nsamples : int, default=None
            Number of samples to average over for the log_prob calculation.
            If provided, then Gaussian errors are assumed, and method will
            look for error columns in `inputs`. Error columns must end in
            `_err`. E.g. the error column for the variable `u` must be `u_err`.
            Zero error assumed for any missing error columns.
        seed : int, default=None
            Random seed for drawing the samples with Gaussian errors.
        returnEnsemble : bool, default=False
            If True, returns log_prob for each flow in the ensemble as an
            array of shape (inputs.shape[0], N flows in ensemble).
            If False, the prob is averaged over the flows in the ensemble,
            and the log of this average is returned as an array of shape
            (inputs.shape[0],)

        Returns
        -------
        np.ndarray
            For shape, see returnEnsemble description above.
        """

        # calculate log_prob for each flow in the ensemble
        ensemble = np.array(
            [flow.log_prob(inputs, nsamples, seed) for flow in self._ensemble.values()]
        )

        # re-arrange so that (axis 0, axis 1) = (inputs, flows in ensemble)
        ensemble = np.rollaxis(ensemble, axis=1)

        if returnEnsemble:
            # return the ensemble of log_probs
            return ensemble
        else:
            # return mean over ensemble
            # note we return log(mean prob) instead of just mean log_prob
            return np.log(np.exp(ensemble).mean(axis=1))

    def posterior(
        self,
        inputs: pd.DataFrame,
        column: str,
        grid: np.ndarray,
        normalize: bool = True,
        nsamples: int = None,
        seed: int = None,
        batch_size: int = None,
        returnEnsemble: bool = False,
    ) -> np.ndarray:
        """Calculates posterior distributions for the provided column.

        Calculates the conditional posterior distribution, assuming the
        data values in the other columns of the DataFrame.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data on which the posterior distributions are conditioned.
            Must have columns matching self.data_columns, *except*
            for the column specified for the posterior (see below).
        column : str
            Name of the column for which the posterior distribution
            is calculated. Must be one of the columns in self.data_columns.
            However, whether or not this column is one of the columns in
            `inputs` is irrelevant.
        grid : np.ndarray
            Grid on which to calculate the posterior.
        normalize : boolean, default=True
            Whether to normalize the posterior so that it integrates to 1.
        nsamples : int, default=None
            Number of samples to average over for the posterior calculation.
            If provided, then Gaussian errors are assumed, and method will
            look for error columns in `inputs`. Error columns must end in
            `_err`. E.g. the error column for the variable `u` must be `u_err`.
            Zero error assumed for any missing error columns.
        seed : int, default=None
            Random seed for drawing the samples with Gaussian errors.
        batch_size : int, default=None
            Size of batches in which to calculate posteriors. If None, all
            posteriors are calculated simultaneously. Simultaneous calculation
            is faster, but memory intensive for large data sets.
        returnEnsemble : bool, default=False
            If True, returns posterior for each flow in the ensemble as an
            array of shape (inputs.shape[0], N flows in ensemble, grid.size).
            If False, the posterior is averaged over the flows in the ensemble,
            and returned as an array of shape (inputs.shape[0], grid.size)

        Returns
        -------
        np.ndarray
            For shape, see returnEnsemble description above.
        """

        # calculate posterior for each flow in the ensemble
        ensemble = np.array(
            [
                flow.posterior(inputs, column, grid, False, nsamples, seed, batch_size)
                for flow in self._ensemble.values()
            ]
        )

        # re-arrange so that (axis 0, axis 1) = (inputs, flows in ensemble)
        ensemble = np.rollaxis(ensemble, axis=1)

        if returnEnsemble:
            # return the ensemble of posteriors
            if normalize:
                ensemble = ensemble.reshape(-1, grid.size)
                ensemble = ensemble / np.trapz(y=ensemble, x=grid).reshape(-1, 1)
                ensemble = ensemble.reshape(inputs.shape[0], -1, grid.size)
            return ensemble
        else:
            # return mean over ensemble
            pdfs = ensemble.mean(axis=1)
            if normalize:
                pdfs = pdfs / np.trapz(y=pdfs, x=grid).reshape(-1, 1)
            return pdfs

    def sample(
        self,
        nsamples: int = 1,
        conditions: pd.DataFrame = None,
        save_conditions: bool = True,
        seed: int = None,
        returnEnsemble: bool = False,
    ) -> pd.DataFrame:
        """Returns samples from the normalizing flow.

        Parameters
        ----------
        nsamples : int, default=1
            The number of samples to be returned, either overall or per flow
            in the ensemble (see returnEnsemble below).
        conditions : pd.DataFrame, optional
            If this is a conditional flow, you must pass conditions for
            each sample. nsamples will be drawn for each row in conditions.
        save_conditions : bool, default=True
            If true, conditions will be saved in the DataFrame of samples
            that is returned.
        seed : int, optional
            Sets the random seed for the samples.
        returnEnsemble : bool, default=False
            If True, nsamples is drawn from each flow in the ensemble.
            If False, nsamples are drawn uniformly from the flows in the ensemble.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of samples.
        """

        if returnEnsemble:
            # return nsamples for each flow in the ensemble
            return pd.concat(
                [
                    flow.sample(nsamples, conditions, save_conditions, seed)
                    for flow in self._ensemble.values()
                ],
                keys=self._ensemble.keys(),
            )
        else:
            # return nsamples drawn uniformly from the flows in the ensemble
            N = int(np.ceil(nsamples / len(self._ensemble)))
            samples = pd.concat(
                [
                    flow.sample(N, conditions, save_conditions, seed)
                    for flow in self._ensemble.values()
                ]
            )
            return samples.sample(nsamples, random_state=seed).reset_index(drop=True)

    def save(self, file: str):
        """Saves the ensemble to a file.

        Pickles the ensemble and saves it to a file that can be passed as
        the `file` argument during flow instantiation.

        WARNING: Currently, this method only works for bijectors that are
        implemented in the `bijectors` module. If you want to save a flow
        with a custom bijector, you either need to add the bijector to that
        module, or handle the saving and loading on your end.

        Parameters
        ----------
        file : str
            Path to where the ensemble will be saved.
            Extension `.pkl` will be appended if not already present.
        """
        save_dict = {name: flow._save_dict() for name, flow in self._ensemble.items()}
        save_dict["class"] = "FlowEnsemble"

        if not file.endswith(".pkl"):
            file += ".pkl"
        with open(file, "wb") as handle:
            pickle.dump(save_dict, handle, recurse=True)

    def train(
        self,
        inputs: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 1024,
        optimizer: Optimizer = None,
        loss_fn: Callable = None,
        sample_errs: bool = False,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict:
        """Trains the normalizing flows on the provided inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data on which to train the normalizing flows.
            Must have columns matching self.data_columns.
        epochs : int, default=50
            Number of epochs to train.
        batch_size : int, default=1024
            Batch size for training.
        optimizer : jax Optimizer, default=adam(step_size=1e-3)
            An optimizer from jax.experimental.optimizers.
        loss_fn : Callable, optional
            A function to calculate the loss: loss = loss_fn(params, x).
            If not provided, will be -mean(log_prob).
        sample_errs : bool, default=False
            Whether to draw new data from the error distributions during
            each epoch of training. Assumes errors are Gaussian, and method
            will look for error columns in `inputs`. Error columns must end
            in `_err`. E.g. the error column for the variable `u` must be
            `u_err`. Zero error assumed for any missing error columns.
        seed : int, default=0
            A random seed to control the batching and the (optional)
            error sampling.
        verbose : bool, default=False
            If true, print the training loss every 5% of epochs.

        Returns
        -------
        dict
            Dictionary of training losses from every epoch for each flow
            in the ensemble.
        """

        loss_dict = dict()

        for name, flow in self._ensemble.items():

            if verbose:
                print(name)

            loss_dict[name] = flow.train(
                inputs,
                epochs,
                batch_size,
                optimizer,
                loss_fn,
                sample_errs,
                seed,
                verbose,
            )

        return loss_dict