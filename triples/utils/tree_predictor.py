import numpy as np

class TreePredictor:
    """
    Light-weight implementation of tree predictor without sklearn dependency.
    Using the compact tree format, `node[i]` has children `node[2*i+1]` and `node[2*i+2]`.

    Supports conversion from:
    XGBRegressor, XGBClassifier, RandomForestRegressor, RandomForestClassifier

    See `from_xgb_model` and `from_random_forest` for more details.

    Limitations in the current implementation:
    * Missing values go to the right child.
    * Predictions must be scalar for a single tree.
    """
    __version__ = '0.0.1'
    __float_type = float
    func = None
    op = None
    def __init__(self, branches=None, thresholds=None, features=None, method='', intercept=0):
        self.branches = branches
        self.thresholds = thresholds
        self.features = features if features is not None else []
        self.method = method
        self.intercept = intercept
        self.__version__ = TreePredictor.__version__

    @property
    def n_estimators(self) -> int:
        return len(self.branches) if self.branches is not None else 0

    @property
    def intercept_(self) -> float:
        return self.intercept

    def __str__(self) -> str:
        return f'TreePredictor(method = "{self.method}", n_estimators = {self.n_estimators})'

    def __repr__(self) -> str:
        return f'<TreePredictor(method = "{self.method}", n_estimators = {self.n_estimators})>'

    @property
    def _float_type(self):
        if hasattr(self.thresholds, 'dtype'):
            return self.thresholds.dtype.type
        return self.__float_type

    def get_default_func(self, func=None):
        if func is not None:
            return func
        if self.func is not None:
            return self.func
        if self.method == 'XGBRegressor':
            return lambda x: float(sum(x) + self.intercept_)
        elif self.method == 'XGBClassifier':
            z = np.log(self.intercept_/(1 - self.intercept_))
            return lambda x: float(1/(1 + np.exp(-(sum(x) + z))))
        elif self.method == 'RandomForestRegressor':
            return lambda x: float(sum(x)/len(x) + self.intercept_)
        elif self.method == 'RandomForestClassifier':
            return lambda x: float(sum(x)/len(x) + self.intercept_)
        return lambda x: float(sum(x) + self.intercept_)

    def get_default_op(self, op=None):
        if op is not None:
            return op
        if self.op is not None:
            return self.op
        if self.method.startswith('XGB'):
            op = lambda v, t: v < t
        else:
            op = lambda v, t: v <= t
        return op

    def predict(self, x: dict, op=None, func=None):
        """
        Predict the output for a single sample.

        Parameters
        ----------
        x : dict
            A dictionary mapping feature names to feature values.
            Missing features will be set to None.
        op : callable, optional
            The operator function to use for comparing the values against thresholds.
            Defaults to `<=` for sklearn and '<' for xgb models.
        func : callable, optional
            The function to use for aggregating the predictions.
            Defaults to the behaviour of XGB / sklearn regressors or binary classifiers.
            Use `lambda x: x` to get prediction (list) of each estimator.

        Returns
        ---------
        float :
            The predicted output.
        """
        preds = [0]*self.n_estimators
        float_type = self._float_type
        x = [float_type(x.get(feature, None)) for feature in self.features]
        op = self.get_default_op(op)
        for i in range(self.n_estimators):
            k = 0
            branches, thresholds = self.branches[i], self.thresholds[i]
            while branches[k] >= 0:
                v = x[branches[k]]
                if v is not None and op(v, thresholds[k]):
                    k = 2*k + 1
                else:
                    k = 2*k + 2
            preds[i] = thresholds[k]
        func = self.get_default_func(func)
        return func(preds)

    def __call__(self, x: dict, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    @classmethod
    def from_xgb_model(cls, xgb_model) -> 'TreePredictor':
        """xgb_model: XGBRegressor or XGBClassifier"""
        model = cls.from_xgb_string(''.join(xgb_model.get_booster().get_dump(dump_format="text")))
        intercept = xgb_model.intercept_.flatten().tolist()[0]
        model.intercept = intercept
        model.method = xgb_model.__class__.__name__
        return model

    @classmethod
    def from_xgb_string(cls, s: str) -> 'TreePredictor':
        """
        Note: xgb model has an `intercept_` parameter, which is not
        stored in the `get_dump()` string. The model should be corrected
        by setting `intercept` manually.
        """
        lines = s.split('\n')
        n_trees = -1
        feature_inds = {}
        varind, threshold = -1, 0.
        trees, nodeids, branches, thresholds = [], [], [], []
        nodemap = {0: 0}

        for line in lines:
            line = line.strip()
            if (not len(line)) or line[0] == 'b': # booster
                continue

            i = line.index(':')
            nodeid = int(line[:i])
            nodeid = nodemap.get(nodeid, nodeid)

            if nodeid == 0:
                n_trees += 1
                nodemap = {0: 0}
            rest = line[i+1:]
            if rest[0] == '[':
                r = rest.index(']')
                l = rest.index('<')
                feature = rest[1:l]
                if feature not in feature_inds:
                    feature_inds[feature] = len(feature_inds)
                varind = feature_inds[feature]
                threshold = float(rest[l+1:r])

                rest = rest[r+1:]
                l, r = rest.index('='), rest.index(',')
                yes = int(rest[l+1:r])
                rest = rest[r+1:]
                l, r = rest.index('='), rest.index(',')
                no = int(rest[l+1:r])
                nodemap[no] = nodeid*2 + 2
                nodemap[yes] = nodeid*2 + 1
            elif rest.startswith('leaf='):
                varind = -1
                threshold = float(rest[5:])
            else:
                raise ValueError(f'Unknown line: {line}')

            trees.append(n_trees)
            nodeids.append(nodeid)
            branches.append(varind)
            thresholds.append(threshold)

        trees = np.array(trees, dtype=int)
        nodeids = np.array(nodeids, dtype=int)
        _branches = np.array(branches, dtype=int)
        _thresholds = np.array(thresholds, dtype=np.float32)

        n_trees = np.max(trees) + 1
        nodes = np.max(nodeids) + 1
        branches = np.full((n_trees * nodes,), -1, dtype=int)
        thresholds = np.zeros((n_trees * nodes,), dtype=np.float32)
        inds = trees * nodes + nodeids

        branches[inds] = _branches
        thresholds[inds] = _thresholds
        branches = branches.reshape((n_trees, nodes))
        thresholds = thresholds.reshape((n_trees, nodes))

        features = [None] * len(feature_inds)
        for k, v in feature_inds.items():
            features[v] = k
        obj = cls(branches, thresholds, features)
        return obj

    @classmethod
    def from_random_forest(cls, skl_model) -> 'TreePredictor':
        """skl_model: RandomForestRegressor or RandomForestClassifier"""
        method = skl_model.__class__.__name__
        length = max([len(e.tree_.children_left) for e in skl_model.estimators_], default=0)
        nodemap = [0] * length
        nodeinv = [0] * length
        branches, thresholds = [None] * skl_model.n_estimators, [None] * skl_model.n_estimators
        flatten = (lambda x: x[:,0,1].flatten()) if 'Classifier' in method else (lambda x: x.flatten())
        for n_tree, estimator in enumerate(skl_model.estimators_):
            tree = estimator.tree_
            children_left = tree.children_left
            children_right = tree.children_right
            branch = tree.feature
            n = len(branch)
            threshold = np.where(branch >= 0, tree.threshold, flatten(tree.value))

            # reorder node indices
            for i in range(length//2):
                if nodeinv[i] < n:
                    j = children_left[nodeinv[i]]
                    k = children_right[nodeinv[i]]
                    nodemap[j] = 2*i+1
                    nodemap[k] = 2*i+2
                    if 2*i + 1 < length:
                        nodeinv[2*i+1] = j
                        if 2*i + 2 < length:
                            nodeinv[2*i+2] = k

            branches[n_tree] = branch[nodeinv]
            thresholds[n_tree] = threshold[nodeinv]
        branches = np.vstack(branches, dtype=int)
        thresholds = np.vstack(thresholds, dtype=np.float64)
        features = skl_model.feature_names_in_.tolist()
        obj = cls(branches, thresholds, features, method=method)
        return obj


    def save_model(self, filename: str) -> 'TreePredictor':
        """
        Save the model to a .npz file.
        NOTE: .npz is safer and more trustable than .pkl.
        """
        if filename.endswith('.npz'):
            attrs = ['branches', 'thresholds', 'features', 'intercept', 'method', '__version__']
            data = {k: getattr(self, k) for k in attrs}
            np.savez(filename, **data)
        else:
            raise ValueError(f"Unknown file extension: {filename}, currently only support .npz.")
        return self

    @classmethod
    def load_model(cls, filename: str) -> 'TreePredictor':
        """Load the model from a .npz file."""
        obj = None
        if filename.endswith('.npz'):
            data = np.load(filename)
            attrs = {'branches', 'thresholds', 'features', 'intercept', 'method', '__version__'}
            scalars = {'intercept', 'method', '__version__'}
            obj = object.__new__(cls)
            for k in attrs:
                if not k in data:
                    raise ValueError(f"Missing key: {k} in {filename}")
                d = data[k]
                if k in scalars and isinstance(d, np.ndarray):
                    d = d.flatten().tolist()[0]
                setattr(obj, k, d)
            if not obj.__version__ == cls.__version__:
                from warnings import warn
                warn(f"Version mismatch: {obj.__version__} != {cls.__version__} in {filename}",
                     category=UserWarning)
        else:
            raise ValueError(f"Unknown file extension: {filename}, currently only support .npz.")
        return obj