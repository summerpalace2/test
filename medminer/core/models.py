"""算法注册表：分类 / 回归 / 聚类。新增算法只需在这里加一项，界面自动生成参数表单。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, KMeans
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .params import ParamSpec, defaults

P = ParamSpec


@dataclass
class ModelSpec:
    key: str
    name: str
    task: str  # classification | regression | clustering
    factory: Callable[..., Any]
    params: list[ParamSpec] = field(default_factory=list)
    needs_scaling: bool = False
    supports_class_weight: bool = False
    tree_based: bool = False
    linear: bool = False
    help: str = ""
    fixed: dict[str, Any] = field(default_factory=dict)
    scale_target: bool = False  # 回归：对目标变量做标准化（SVR/MLP 对目标量级敏感），预测时自动还原

    def build(self, params: dict | None = None, class_weight: str | None = None, seed: int = 42):
        kw = {**defaults(self.params), **(params or {}), **self.fixed}
        for s in self.params:
            if s.name in kw:
                kw[s.name] = s.coerce(kw[s.name])
        if class_weight and self.supports_class_weight:
            kw["class_weight"] = class_weight
        try:
            return self.factory(random_state=seed, **kw)
        except TypeError:  # 无 random_state 的算法
            return self.factory(**kw)

    def grid(self) -> dict[str, list]:
        return {s.name: s.grid for s in self.params if s.grid}


def _xgb_cls(**kw):
    from xgboost import XGBClassifier
    return XGBClassifier(eval_metric="logloss", verbosity=0, n_jobs=1, **kw)


def _xgb_reg(**kw):
    from xgboost import XGBRegressor
    return XGBRegressor(verbosity=0, n_jobs=1, **kw)


def _lgbm_cls(**kw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(verbose=-1, n_jobs=1, **kw)


def _lgbm_reg(**kw):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(verbose=-1, n_jobs=1, **kw)


_n_est = lambda d=200: P("n_estimators", "树的数量", "int", d, 10, 2000, 10, grid=[100, 200, 400])
_depth = lambda d=None: P("max_depth", "最大深度", "int", d, 1, 50, 1, nullable=True, grid=[3, 5, 8, None] if d is None else [2, 3, 4, 6])
_lr = lambda: P("learning_rate", "学习率", "float", 0.1, 0.001, 1.0, 0.01, grid=[0.03, 0.1, 0.3])
_minleaf = lambda: P("min_samples_leaf", "叶节点最小样本数", "int", 1, 1, 100, 1, grid=[1, 3, 5, 10])

CLASSIFIERS = [
    ModelSpec("logreg", "逻辑回归", "classification", LogisticRegression,
              [P("C", "正则强度倒数 C", "float", 1.0, 0.0001, 1000, 0.1, grid=[0.01, 0.1, 1, 10]),
               P("penalty", "正则项", "choice", "l2", choices=["l2", "none"]),
               P("max_iter", "最大迭代", "int", 1000, 100, 10000, 100)],
              needs_scaling=True, supports_class_weight=True, linear=True,
              help="可解释性最强的基线模型，系数可换算为 OR。"),
    ModelSpec("dtree", "决策树", "classification", DecisionTreeClassifier,
              [_depth(4), _minleaf(), P("criterion", "划分准则", "choice", "gini", choices=["gini", "entropy"])],
              supports_class_weight=True, tree_based=True),
    ModelSpec("rf", "随机森林", "classification", RandomForestClassifier,
              [_n_est(300), _depth(), _minleaf(), P("max_features", "每次划分特征数", "choice", "sqrt", choices=["sqrt", "log2"])],
              supports_class_weight=True, tree_based=True, fixed={"n_jobs": 1}),
    ModelSpec("gb", "梯度提升树 (sklearn)", "classification", GradientBoostingClassifier,
              [_n_est(150), _lr(), P("max_depth", "最大深度", "int", 3, 1, 10, 1, grid=[2, 3, 4])], tree_based=True),
    ModelSpec("xgb", "XGBoost", "classification", _xgb_cls,
              [_n_est(200), _lr(), P("max_depth", "最大深度", "int", 3, 1, 12, 1, grid=[2, 3, 5]),
               P("subsample", "行采样比例", "float", 0.9, 0.3, 1, 0.1)], tree_based=True),
    ModelSpec("lgbm", "LightGBM", "classification", _lgbm_cls,
              [_n_est(200), _lr(), P("num_leaves", "叶子数", "int", 15, 2, 256, 1, grid=[7, 15, 31]),
               P("min_child_samples", "叶最小样本", "int", 10, 1, 200, 1)], tree_based=True),
    ModelSpec("svm", "支持向量机 (RBF)", "classification", SVC,
              [P("C", "惩罚系数 C", "float", 1.0, 0.001, 1000, 0.5, grid=[0.1, 1, 10]),
               P("gamma", "核系数 gamma", "choice", "scale", choices=["scale", "auto"]),
               P("kernel", "核函数", "choice", "rbf", choices=["rbf", "linear", "poly", "sigmoid"])],
              needs_scaling=True, supports_class_weight=True, fixed={"probability": True}),
    ModelSpec("knn", "K 近邻", "classification", KNeighborsClassifier,
              [P("n_neighbors", "邻居数 K", "int", 7, 1, 100, 1, grid=[3, 5, 7, 11, 15]),
               P("weights", "权重", "choice", "uniform", choices=["uniform", "distance"])], needs_scaling=True),
    ModelSpec("nb", "高斯朴素贝叶斯", "classification", GaussianNB, [P("var_smoothing", "方差平滑", "float", 1e-9, 1e-12, 1e-3, 1e-9)]),
    ModelSpec("adaboost", "AdaBoost", "classification", AdaBoostClassifier,
              [P("n_estimators", "弱学习器数", "int", 100, 10, 1000, 10), P("learning_rate", "学习率", "float", 0.5, 0.01, 2, 0.05)],
              tree_based=True),
    ModelSpec("mlp", "多层感知机", "classification", MLPClassifier,
              [P("hidden_layer_sizes", "隐层神经元数", "int", 32, 2, 512, 2), P("alpha", "L2 正则", "float", 0.001, 0, 1, 0.001),
               P("max_iter", "最大迭代", "int", 800, 100, 5000, 100)], needs_scaling=True),
]

REGRESSORS = [
    ModelSpec("linreg", "线性回归", "regression", LinearRegression, [], needs_scaling=True, linear=True),
    ModelSpec("ridge", "岭回归", "regression", Ridge, [P("alpha", "正则强度 α", "float", 1.0, 0, 1000, 0.5, grid=[0.1, 1, 10, 100])],
              needs_scaling=True, linear=True),
    ModelSpec("lasso", "Lasso", "regression", Lasso, [P("alpha", "正则强度 α", "float", 0.1, 0.0001, 100, 0.05, grid=[0.01, 0.1, 1])],
              needs_scaling=True, linear=True),
    ModelSpec("enet", "弹性网络", "regression", ElasticNet,
              [P("alpha", "正则强度 α", "float", 0.1, 0.0001, 100, 0.05, grid=[0.01, 0.1, 1]),
               P("l1_ratio", "L1 比例", "float", 0.5, 0, 1, 0.1, grid=[0.2, 0.5, 0.8])], needs_scaling=True, linear=True),
    ModelSpec("dtree_r", "决策树", "regression", DecisionTreeRegressor, [_depth(4), _minleaf()], tree_based=True),
    ModelSpec("rf_r", "随机森林", "regression", RandomForestRegressor, [_n_est(300), _depth(), _minleaf()],
              tree_based=True, fixed={"n_jobs": 1}),
    ModelSpec("gb_r", "梯度提升树 (sklearn)", "regression", GradientBoostingRegressor,
              [_n_est(150), _lr(), P("max_depth", "最大深度", "int", 3, 1, 10, 1, grid=[2, 3, 4])], tree_based=True),
    ModelSpec("xgb_r", "XGBoost", "regression", _xgb_reg,
              [_n_est(200), _lr(), P("max_depth", "最大深度", "int", 3, 1, 12, 1, grid=[2, 3, 5])], tree_based=True),
    ModelSpec("lgbm_r", "LightGBM", "regression", _lgbm_reg,
              [_n_est(200), _lr(), P("num_leaves", "叶子数", "int", 15, 2, 256, 1, grid=[7, 15, 31])], tree_based=True),
    ModelSpec("svr", "支持向量回归", "regression", SVR,
              [P("C", "惩罚系数 C", "float", 1.0, 0.001, 1000, 0.5, grid=[0.1, 1, 10]),
               P("kernel", "核函数", "choice", "rbf", choices=["rbf", "linear", "poly"])], needs_scaling=True,
              scale_target=True),
    ModelSpec("knn_r", "K 近邻", "regression", KNeighborsRegressor,
              [P("n_neighbors", "邻居数 K", "int", 7, 1, 100, 1, grid=[3, 5, 7, 11, 15])], needs_scaling=True),
    ModelSpec("mlp_r", "多层感知机", "regression", MLPRegressor,
              [P("hidden_layer_sizes", "隐层神经元数", "int", 32, 2, 512, 2), P("alpha", "L2 正则", "float", 0.001, 0, 1, 0.001),
               P("max_iter", "最大迭代", "int", 1000, 100, 5000, 100)], needs_scaling=True, scale_target=True),
]


CLUSTERERS = [
    ModelSpec("kmeans", "K-Means", "clustering", KMeans,
              [P("n_clusters", "簇数 K", "int", 3, 2, 30, 1), P("n_init", "初始化次数", "int", 10, 1, 100, 1)]),
    ModelSpec("agglo", "层次聚类", "clustering", AgglomerativeClustering,
              [P("n_clusters", "簇数 K", "int", 3, 2, 30, 1),
               P("linkage", "连接方式", "choice", "ward", choices=["ward", "complete", "average", "single"])]),
    ModelSpec("gmm", "高斯混合模型", "clustering", GaussianMixture,
              [P("n_components", "成分数", "int", 3, 2, 30, 1),
               P("covariance_type", "协方差类型", "choice", "full", choices=["full", "tied", "diag", "spherical"])]),
    ModelSpec("dbscan", "DBSCAN", "clustering", DBSCAN,
              [P("eps", "邻域半径 eps", "float", 1.5, 0.05, 50, 0.1), P("min_samples", "最小样本数", "int", 5, 1, 100, 1)]),
    ModelSpec("birch", "BIRCH", "clustering", Birch, [P("n_clusters", "簇数 K", "int", 3, 2, 30, 1),
                                                      P("threshold", "阈值", "float", 0.5, 0.05, 10, 0.1)]),
]

REGISTRY: dict[str, ModelSpec] = {m.key: m for m in CLASSIFIERS + REGRESSORS + CLUSTERERS}


def models_for(task: str) -> list[ModelSpec]:
    return [m for m in REGISTRY.values() if m.task == task]


def build_estimator(spec: ModelSpec, params: dict | None = None, class_weight: str | None = None, seed: int = 42):
    """MLP 的 hidden_layer_sizes 在界面里是单个整数，这里转成元组。"""
    params = dict(params or {})
    if "hidden_layer_sizes" in params and not isinstance(params["hidden_layer_sizes"], (tuple, list)):
        params["hidden_layer_sizes"] = (int(params["hidden_layer_sizes"]),)
    est = spec.build({k: v for k, v in params.items() if k != "hidden_layer_sizes"}, class_weight, seed)
    if "hidden_layer_sizes" in params:
        est.set_params(hidden_layer_sizes=params["hidden_layer_sizes"])
    if spec.scale_target:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler
        est = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    return est
