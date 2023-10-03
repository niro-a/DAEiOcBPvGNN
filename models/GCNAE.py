# Code adapted from the PyGOD 1.0.0 library. Url: https://docs.pygod.org/
# Original Authors: Yue Zhao <zhaoy@cmu.edu>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause


import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN
from torch_geometric.utils import to_dense_adj

from pygod.nn.decoder import DotProductDecoder
import numpy as np

class GAEBase(nn.Module):
    """
    Graph Autoencoder

    See :cite:`kipf2016variational` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    recon_s : bool, optional
        Reconstruct the structure instead of node feature .
        Default: ``False``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 encoder_layers=3,
                 decoder_layers=1,
                 dropout=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,
                 loss_func=F.mse_loss,
                 **kwargs):
        super(GAEBase, self).__init__()

        self.backbone = backbone
        # split the number of layers for the encoder and decoders
        assert encoder_layers+decoder_layers >= 2, \
            "Number of layers must be greater than or equal to 2."

        self.encoder = self.backbone(in_channels=in_dim,
                                     hidden_channels=hid_dim,
                                     out_channels=hid_dim,
                                     num_layers=encoder_layers,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)
        self.recon_s = recon_s
        if self.recon_s:
            self.decoder = DotProductDecoder(in_dim=hid_dim,
                                             hid_dim=hid_dim,
                                             num_layers=decoder_layers,
                                             dropout=dropout,
                                             act=act,
                                             sigmoid_s=sigmoid_s,
                                             backbone=self.backbone,
                                             **kwargs)
        else:
            self.decoder = self.backbone(in_channels=hid_dim,
                                         hidden_channels=hid_dim,
                                         out_channels=in_dim,
                                         num_layers=decoder_layers,
                                         dropout=dropout,
                                         act=act,
                                         **kwargs)

        self.emb = None

    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed embeddings.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        edge_index = edge_index.to(device)

        if self.backbone == MLP:
            self.emb = self.encoder(x, None)
            x_ = self.decoder(self.emb, None)
        else:
            self.emb = self.encoder(x, edge_index)
            x_ = self.decoder(self.emb, edge_index)
        return x_

    @staticmethod
    def process_graph(data, recon_s=False):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        recon_s : bool, optional
            Reconstruct the structure instead of node feature .
        """
        if recon_s:
            data.s = to_dense_adj(data.edge_index)[0]


import torch
import warnings
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN

import time
from inspect import signature
from abc import ABC, abstractmethod

import torch
import numpy as np
from scipy.stats import binom
from scipy.special import erf

from torch_geometric.nn import GIN
from torch_geometric import compile
from torch_geometric.loader import NeighborLoader

from pygod.utils import logger, validate_device, pprint, is_fitted


class Detector(ABC):
    """Abstract class for all outlier detection algorithms.

    Parameters
    ----------
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N`*``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.

    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    """

    def __init__(self,
                 contamination=0.1,
                 verbose=0):

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination
        self.verbose = verbose
        self.decision_score_ = None

    @abstractmethod
    def process_graph(self, data):
        """
        Data preprocessing for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph.
        """

    @abstractmethod
    def fit(self, data, label=None):
        """Fit detector with training data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The training graph.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted detector.
        """

    @abstractmethod
    def decision_function(self, data, label=None):
        """Predict raw outlier scores of testing data using the fitted
        detector. Outliers are assigned with higher outlier scores.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The testing graph.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.

        Returns
        -------
        score : torch.Tensor
            The outlier scores of shape :math:`N`.
        """

    def predict(self,
                data=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False):
        """Prediction for testing data using the fitted detector.
        Return predicted labels by default.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            The testing graph. If ``None``, the training data is used.
            Default: ``None``.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.
        return_pred : bool, optional
            Whether to return the predicted binary labels. The labels
            are determined by the outlier contamination on the raw
            outlier scores. Default: ``True``.
        return_score : bool, optional
            Whether to return the raw outlier scores.
            Default: ``False``.
        return_prob : bool, optional
            Whether to return the outlier probabilities.
            Default: ``False``.
        prob_method : str, optional
            The method to convert the outlier scores to probabilities.
            Two approaches are possible:

            1. ``'linear'``: simply use min-max conversion to linearly
            transform the outlier scores into the range of
            [0,1]. The model must be fitted first.

            2. ``'unify'``: use unifying scores,
            see :cite:`kriegel2011interpreting`.

            Default: ``'linear'``.
        return_conf : boolean, optional
            Whether to return the model's confidence in making the same
            prediction under slightly different training sets.
            See :cite:`perini2020quantifying`. Default: ``False``.

        Returns
        -------
        pred : torch.Tensor
            The predicted binary outlier labels of shape :math:`N`.
            0 stands for inliers and 1 for outliers.
            Only available when ``return_label=True``.
        score : torch.Tensor
            The raw outlier scores of shape :math:`N`.
            Only available when ``return_score=True``.
        prob : torch.Tensor
            The outlier probabilities of shape :math:`N`.
            Only available when ``return_prob=True``.
        conf : torch.Tensor
            The prediction confidence of shape :math:`N`.
            Only available when ``return_conf=True``.
        """

        is_fitted(self, ['decision_score_', 'threshold_', 'label_'])

        output = ()
        if data is None:
            score = self.decision_score_
            logger(score=self.decision_score_,
                   target=label,
                   verbose=self.verbose,
                   train=False)
        else:
            score = self.decision_function(data, label)
        if return_pred:
            pred = (score > self.threshold_).long()
            output += (pred,)
        if return_score:
            output += (score,)
        if return_prob:
            prob = self._predict_prob(score, prob_method)
            output += (prob,)
        if return_conf:
            conf = self._predict_conf(score)
            output += (conf,)

        if len(output) == 1:
            return output[0]
        else:
            return output

    def _predict_prob(self, score, method='linear'):
        """Predict the probabilities of being outliers. Two approaches
        are possible:

        'linear': simply use min-max conversion to linearly
                  transform the outlier scores into the range of
                  [0,1]. The model must be fitted first.

        'unify': use unifying scores,
                 see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        score : torch.Tensor
            The outlier scores of shape :math:`N`.

        method : str
            probability conversion method. It must be one of
            'linear' or 'unify'. Default: ``linear``.

        Returns
        -------
        prob : torch.Tensor
            The outlier probabilities of shape :math:`N`.
        """

        if method == 'linear':
            train_score = self.decision_score_
            prob = score - train_score.min()
            prob /= train_score.max() - train_score.min()
            prob = prob.clamp(0, 1)
        elif method == 'unify':
            mu = torch.mean(self.decision_score_)
            sigma = torch.std(self.decision_score_)
            pre_erf_score = (score - mu) / (sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            prob = erf_score.clamp(0, 1)
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')
        return prob

    def _predict_conf(self, score):
        """Predict the model's confidence in making the same prediction
        under slightly different training sets.
        See :cite:`perini2020quantifying`.

        Parameters
        ----------
        score : torch.Tensor
            The outlier score of shape :math:`N`.

        Returns
        -------
        conf : torch.Tensor
            The prediction confidence of shape :math:`N`.
        """

        n = len(self.decision_score_)
        k = n - int(n * self.contamination)

        n_ins = (self.decision_score_.view(n, 1) <= score).count_nonzero(dim=0)

        # Derive the outlier probability using Bayesian approach
        post_prob = (1 + n_ins) / (2 + n)

        # Transform the outlier probability into a confidence value
        conf = torch.Tensor(1 - binom.cdf(k, n, post_prob))

        pred = (score > self.threshold_).long()
        conf = torch.where(pred == 0, 1 - conf, conf)
        return conf

    def _process_decision_score(self):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - label_: binary labels of training data
        """

        self.threshold_ = np.percentile(self.decision_score_,
                                        100 * (1 - self.contamination))
        self.label_ = (self.decision_score_ > self.threshold_).long()

    def __repr__(self):

        class_name = self.__class__.__name__
        init_signature = signature(self.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        params = {}
        for key in sorted([p.name for p in parameters]):
            params[key] = getattr(self, key, None)
        return '%s(%s)' % (class_name, pprint(params, offset=len(class_name)))


class DeepDetector(Detector, ABC):
    """
    Abstract class for deep outlier detection algorithms.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GIN``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    gan : bool, optional
        Whether using adversarial training. Default: ``False``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs
        Other parameters for the backbone.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N`*``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GIN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 gan=False,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(DeepDetector, self).__init__(contamination=contamination,
                                           verbose=verbose)

        # model param
        self.in_dim = None
        self.num_nodes = None
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.backbone = backbone
        self.kwargs = kwargs

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = gpu
        self.batch_size = batch_size
        self.gan = gan
        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

        # other param
        self.model = None
        self.save_emb = save_emb
        if save_emb:
            self.emb = None
        self.compile_model = compile_model

    def fit(self, data, label=None):

        self.num_nodes, self.in_dim = data.x.shape
        self.process_graph(data)
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model = self.init_model(**self.kwargs)
        self.model = self.model.to(self.device)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_g = torch.optim.Adam(self.model.generator.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.model.discriminator.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.epoch):
            start_time = time.time()
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_g = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                loss, score = self.forward_model(sampled_data.to(self.device))
                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) == tuple:
                        self.emb[0][node_idx[:batch_size]] = \
                            self.model.emb[0][:batch_size].cpu()
                        self.emb[1][node_idx[:batch_size]] = \
                            self.model.emb[1][:batch_size].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = \
                            self.model.emb[:batch_size].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_value = epoch_loss / data.x.shape[0]
            if self.gan:
                loss_value = (self.epoch_loss_g / data.x.shape[0], loss_value)
            logger(epoch=epoch,
                   loss=loss_value,
                   score=self.decision_score_,
                   target=label,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

        self._process_decision_score()
        return self

    def decision_function(self, data, label=None):

        self.process_graph(data)
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_score = torch.zeros(data.x.shape[0])
        if self.save_emb:
            if type(self.hid_dim) is tuple:
                self.emb = (torch.zeros(data.x.shape[0], self.hid_dim[0]),
                            torch.zeros(data.x.shape[0], self.hid_dim[1]))
            else:
                self.emb = torch.zeros(data.x.shape[0], self.hid_dim)
        start_time = time.time()
        for sampled_data in loader:
            loss, score = self.forward_model(sampled_data)
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id
            if self.save_emb:
                if type(self.hid_dim) is tuple:
                    self.emb[0][node_idx[:batch_size]] = \
                        self.model.emb[0][:batch_size].cpu()
                    self.emb[1][node_idx[:batch_size]] = \
                        self.model.emb[1][:batch_size].cpu()
                else:
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()

            outlier_score[node_idx[:batch_size]] = score

        logger(loss=loss.item() / data.x.shape[0],
               score=outlier_score,
               target=label,
               time=time.time() - start_time,
               verbose=self.verbose,
               train=False)
        return outlier_score

    def predict(self,
                data=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False,
                return_emb=False):
        """Prediction for testing data using the fitted detector.
        Return predicted labels by default.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            The testing graph. If ``None``, the training data is used.
            Default: ``None``.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.
        return_pred : bool, optional
            Whether to return the predicted binary labels. The labels
            are determined by the outlier contamination on the raw
            outlier scores. Default: ``True``.
        return_score : bool, optional
            Whether to return the raw outlier scores.
            Default: ``False``.
        return_prob : bool, optional
            Whether to return the outlier probabilities.
            Default: ``False``.
        prob_method : str, optional
            The method to convert the outlier scores to probabilities.
            Two approaches are possible:

            1. ``'linear'``: simply use min-max conversion to linearly
            transform the outlier scores into the range of
            [0,1]. The model must be fitted first.

            2. ``'unify'``: use unifying scores,
            see :cite:`kriegel2011interpreting`.

            Default: ``'linear'``.
        return_conf : boolean, optional
            Whether to return the model's confidence in making the same
            prediction under slightly different training sets.
            See :cite:`perini2020quantifying`. Default: ``False``.
        return_emb : bool, optional
            Whether to return the learned node representations.
            Default: ``False``.

        Returns
        -------
        pred : torch.Tensor
            The predicted binary outlier labels of shape :math:`N`.
            0 stands for inliers and 1 for outliers.
            Only available when ``return_label=True``.
        score : torch.Tensor
            The raw outlier scores of shape :math:`N`.
            Only available when ``return_score=True``.
        prob : torch.Tensor
            The outlier probabilities of shape :math:`N`.
            Only available when ``return_prob=True``.
        conf : torch.Tensor
            The prediction confidence of shape :math:`N`.
            Only available when ``return_conf=True``.
        """
        if return_emb:
            self.save_emb = True

        output = super(DeepDetector, self).predict(data,
                                                   label,
                                                   return_pred,
                                                   return_score,
                                                   return_prob,
                                                   prob_method,
                                                   return_conf)
        if return_emb:
            if type(output) == tuple:
                output += (self.emb,)
            else:
                output = (output, self.emb)

        return output

    @abstractmethod
    def init_model(self):
        """
        Initialize the neural network detector.

        Returns
        -------
        model : torch.nn.Module
            The initialized neural network detector.
        """

    @abstractmethod
    def forward_model(self, data):
        """
        Forward pass of the neural network detector.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph.

        Returns
        -------
        loss : torch.Tensor
            The loss of the current batch.
        score : torch.Tensor
            The outlier scores of the current batch.
        """


class GCNAE(DeepDetector):
    """
    Graph Autoencoder

    See :cite:`kipf2016variational` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    recon_s : bool, optional
        Reconstruct the structure instead of node feature .
        Default: ``False``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs : optional
        Other parameters for the backbone.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N`*``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """

    def __init__(self,
                 hid_dim=64,
                 encoder_layers=3,
                 decoder_layers=1,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 save_emb=False,
                 compile_model=False,
                 loss_func=F.mse_loss,
                 attribute_dims=[],
                 **kwargs):

        if num_neigh != 0 and backbone == MLP:
            warnings.warn('MLP does not use neighbor information.')
            num_neigh = 0

        self.recon_s = recon_s
        self.sigmoid_s = sigmoid_s
        self.attribute_dims = attribute_dims
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        super(GCNAE, self).__init__(hid_dim=hid_dim,
                                  encoder_layers=encoder_layers,
                                  decoder_layers=decoder_layers,
                                  dropout=dropout,
                                  weight_decay=weight_decay,
                                  act=act,
                                  backbone=backbone,
                                  contamination=contamination,
                                  lr=lr,
                                  epoch=epoch,
                                  gpu=gpu,
                                  batch_size=batch_size,
                                  num_neigh=num_neigh,
                                  verbose=verbose,
                                  save_emb=save_emb,
                                  compile_model=compile_model,
                                  loss_func=loss_func,
                                  **kwargs)

    def process_graph(self, data):
        GAEBase.process_graph(data, recon_s=self.recon_s)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)

        kwargs.pop('encoder_layers', None)
        kwargs.pop('decoder_layers', None)

        return GAEBase(in_dim=self.in_dim,
                       hid_dim=self.hid_dim,
                       encoder_layers=self.encoder_layers,
                       decoder_layers=self.decoder_layers,
                       dropout=self.dropout,
                       act=self.act,
                       recon_s=self.recon_s,
                       sigmoid_s=self.sigmoid_s,
                       backbone=self.backbone,
#                       loss_func=self.loss_func,
                       **kwargs)


    def forward_model(self, data):

        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        if self.recon_s:
            s = data.s.to(self.device)[:, node_idx]

        h = self.model(x, edge_index)

        target = s if self.recon_s else x

        def mean_attr_dim(score, attribute_dims):
            start = 0
            score_list = []
            attribute_dims_cumsum = np.cumsum(attribute_dims)
            for end in attribute_dims_cumsum:
                score_slice = score[:, start:end]
                score_mean = torch.mean(score_slice, dim=1)
                score_list.append(score_mean)
                start = end
            return torch.stack(score_list, dim=1)

        score = torch.mean(F.mse_loss(target[:batch_size],
                                                h[:batch_size],
                                                reduction='none'), dim=1)

        dec_score = mean_attr_dim(F.mse_loss(target[:batch_size],
                                             h[:batch_size],
                                             reduction='none'), self.attribute_dims)

        dec_score = torch.mean(dec_score, dim=1)

        loss = torch.mean(score)

        return loss, dec_score.detach().cpu()
