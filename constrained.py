import numpy as np
import tensorflow as tf
import keras
from keras import losses, layers, optimizers, initializers, constraints, Model, callbacks, Input


# Model inputs >>>
"""
Segments: [Mortgage, Construction, Corporate, Consumer, MM]
"""
parameters = {

    # Model inputs >>>
    'im':           [2.50, 5.50, 3.50, 6.25, 0.15],     # Interest margin, % to assets
    'cta':          [0.75, 1.25, 1.00, 1.50, 0.05],     # Cost-to-assets, % to assets
    'el':           [0.50, 1.50, 0.75, 1.75, 0.00],     # Expected loss, % to assets
    'el_std':       [0.10, 1.00, 0.20, 0.75, 0.05],     # Standard deviation of EL, %
    'el_corr':      [                                   # Correlation of EL, %
                    [1.00, 0.80, 0.50, 0.70, 0.10],
                    [0.80, 1.00, 0.70, 0.60, 0.20],
                    [0.50, 0.70, 1.00, 0.50, 0.30],
                    [0.70, 0.60, 0.50, 1.00, 0.20],
                    [0.10, 0.20, 0.30, 0.20, 1.00]],
    'rw':           [75, 150, 100, 150, 10],            # Risk weight, %

    # Model constraints >>>
    'exposure_min': [100, 20, 10, 20, 0],               # Min exposure, US$ bn
    'soft_limit':   [False, False, True, True, False],  # True = "soft" limit of form "greater than or zero"
    'exposure_max': [300, 50, 200, 150, 800],           # Max exposure, US$ bn
    'concentration_min': 0.6,                           # Share of mortgage and construction loans, % of loan book
    't1': 24.0,                                         # Tier 1 Capital, US$ bn
    't2': 4.0,                                          # Tier 2 Capital, US$ bn
    't1_min': 0.060,                                    # Tier 1 Capital Ratio
    'tc_min': 0.080,                                    # Total Capital Ratio
    'ccb_min': 0.025,                                   # Capital Conservation Buffer
    'leverage_min': 0.03,                               # Leverage Ratio
    'roe_min': 15.00,                                   # Return-on-equity threshold, %
    'cor_target': 0.75,                                 # Cost of risk target
    'el_max': 1.25,                                     # Cost of risk limit

    # Model hyper-parameters
    'batch_size': 512,
    'objective': 'raroc',                               # Choose from ['raroc', 'roe', 'risk', 'el']
    'mode': 'max',                                      # Chose from ['max', 'min']
    'drop': [],
    'beta': 1.0,
    'learning_rate': 1.0,
    'decay_rate': 0.95,
    'decay_steps': 10000
 }


# Classes >>>

class Dense2D(layers.Layer):
    """
    """
    def __init__(self, batch_size: int, t1: float, t1_min: float, rw: list, name: str = 'dense2d', **kwargs):
        super(Dense2D, self).__init__(name=name)
        self.batch_size = batch_size
        self.t1 = t1
        self.t1_min = t1_min
        self.rw = np.array(rw) / 100.0

        # Set shape
        shape = (self.batch_size, 5)

        # Random initialization of allocations
        t1 = self.t1 * tf.random.uniform(shape=(), minval=0.1, maxval=1.2)
        allocation = tf.random.uniform(shape=shape, maxval=1.0)
        allocation = allocation / tf.reduce_sum(allocation, axis=-1, keepdims=True) * t1
        exposure = allocation / self.t1_min / self.rw

        # Hard constraints
        min_val = np.array(parameters['exposure_min']) * (1 - np.array(parameters['soft_limit']))
        min_val = tf.constant(min_val, dtype=tf.float32)
        min_val = tf.ones_like(exposure) * tf.expand_dims(min_val, axis=0)
        max_val = tf.constant(parameters['exposure_max'], dtype=tf.float32)
        max_val = tf.ones_like(exposure) * tf.expand_dims(max_val, axis=0)
        exposure_constraints = ExposureConstraint(min_value=min_val, max_value=max_val)

        # Initialize weights with Non-negative constraint
        self.exposure = self.add_weight(
            shape=(self.batch_size, 5),
            initializer=initializers.Constant(exposure),
            constraint=exposure_constraints,
            trainable=True)

    def build(self, input_shape):
        """
        Randomly allocate up to 10% of Tier 1 Capital between asset classes.
        We use random initialization to test convergence to global optima.
        """

    def call(self, inputs,*args, **kwargs):
        return tf.multiply(inputs, self.exposure)

    def compute_output_shape(self, input_shape, *args, **kwargs):
        output_shape = (input_shape[0], 5)
        return output_shape


class AllocationModel(Model):
    """
    """
    def __init__(self,  name='model', **kwargs):
        super(AllocationModel, self).__init__(name=name)
        self.dense = Dense2D(**parameters)

    def call(self, inputs,*args, **kwargs):
        x = self.dense(inputs)
        return x

    def train_step(self, data):
        inputs, y_true = data

        with tf.GradientTape() as tape:

            # Get predictions
            y_pred = self(inputs, training=True)

            # Compute vector loss
            loss = self.loss(y_true, y_pred)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply gradients
        for g, v in zip(gradients, self.trainable_variables):
            if g is not None:
                self.optimizer.apply_gradients([(g, v)])

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}


class PortfolioMetrics:
    """
    Compute metrics and ratios per SAMPLE given exposure per asset class:
        1) Risk-adjusted ROE,
        2) Return on equity,
        3) Risk (diversification ratio),
        4) Expected loss, mean,
        5) Tier 1 Capital ratio,
        6) Total Capital ratio,
        7) Capital Conservation Buffer ratio,
        7) Leverage ratio,
        8) Mortgage and construction segment, % of loan book.
    Make feasibility test and return mask.
    """
    def __init__(self, drop: list, im: list, el: list, el_std: list, cta: list, rw: list, el_corr: list,
                 t1: float, t2: float, cor_target: float,
                 t1_min: float, tc_min:float, ccb_min: float, roe_min: float, leverage_min: float, el_max: float,
                 exposure_min: list, soft_limit: list, exposure_max: list, concentration_min: float, **kwargs):
        self.im = tf.expand_dims(tf.constant(im, dtype=tf.float32), axis=0)
        self.el = tf.expand_dims(tf.constant(el, dtype=tf.float32), axis=0)
        self.el_std = tf.expand_dims(tf.constant(el_std, dtype=tf.float32), axis=0)
        self.cta = tf.expand_dims(tf.constant(cta, dtype=tf.float32), axis=0)
        self.rw = tf.expand_dims(tf.constant(rw, dtype=tf.float32), axis=0) / 100.0
        self.el_corr = tf.expand_dims(tf.constant(el_corr, dtype=tf.float32), axis=0)
        self.t1 = tf.constant(t1, dtype=tf.float32)
        self.t2 = tf.constant(t2, dtype=tf.float32)
        self.t1_min = tf.constant(t1_min, dtype=tf.float32)
        self.tc_min = tf.constant(tc_min, dtype=tf.float32)
        self.soft_limit = tf.where(soft_limit)[:, 0]
        self.cor_target = tf.constant(cor_target, dtype=tf.float32)
        self.indicators = {
            'roe': {'mode': 'min', 'threshold': tf.constant(roe_min, dtype=tf.float32)},
            't1_ratio': {'mode': 'min', 'threshold': tf.constant(t1_min, dtype=tf.float32)},
            'tc_ratio': {'mode': 'min', 'threshold': tf.constant(tc_min, dtype=tf.float32)},
            'ccb_ratio': {'mode': 'min', 'threshold': tf.constant(ccb_min, dtype=tf.float32)},
            'leverage': {'mode': 'min', 'threshold': tf.constant(leverage_min, dtype=tf.float32)},
            'el': {'mode': 'max', 'threshold': tf.constant(el_max, dtype=tf.float32)},
            'exposure_min': {'mode': 'soft', 'threshold': tf.constant(exposure_min, dtype=tf.float32)},
            'exposure_max': {'mode': 'max', 'threshold': tf.constant(exposure_max, dtype=tf.float32)},
            'concentration': {'mode': 'min', 'threshold': tf.constant(concentration_min, dtype=tf.float32)}}
        for x in drop:
            self.indicators.pop(x)
        self.epsilon = 1e-07

    def call(self, exposure):

        # Cast exposure to float32
        exposure = tf.cast(exposure, dtype=tf.float32)

        # Compute metrics
        el, roe = self.el_roe(exposure=exposure)
        mrtg, cstn, corp, cnsr, mmkt = tf.unstack(tf.expand_dims(exposure, axis=-1), axis=1)
        risk = self.risk(exposure=exposure)
        raroc = roe / risk
        t1_ratio = self.t1_ratio(exposure=exposure)
        tc_ratio = self.tc_ratio(exposure=exposure)
        ccb_ratio = tf.minimum(t1_ratio - self.t1_min, tc_ratio - self.tc_min)
        leverage = self.leverage(exposure=exposure)
        concentration = tf.divide(
            tf.reduce_sum(exposure[:, :2], axis=-1, keepdims=True),
            tf.reduce_sum(exposure[:, :-1], axis=-1, keepdims=True) + self.epsilon)
        metrics = {
            'mrtg': mrtg, 'cstn': cstn, 'corp': corp, 'cnsr': cnsr, 'mmkt': mmkt,
            'exposure_max': exposure, 'exposure_min': exposure,
            'raroc': raroc, 'roe': roe, 'risk': risk, 'el': el,
            't1_ratio': t1_ratio, 'tc_ratio': tc_ratio, 'ccb_ratio': ccb_ratio, 'leverage': leverage,
            'concentration': concentration}

        # Check feasibility
        indicators = [self.is_feasible(metric=metrics[x], **self.indicators[x]) for x in self.indicators.keys()]
        indicators = tf.concat(indicators, axis=-1)
        indicators = tf.cast(tf.reduce_prod(indicators, axis=-1), dtype=tf.bool)
        metrics['feasibility'] = indicators

        return metrics

    def el_roe(self, exposure):
        """
        Compute portfolio metrics:
        1) Expected loss,
        2) Return on equity.
        Note that expected loss calculation does not include money market segment.
        This aligns with the banking reporting practice where MM assets are reported at market value.
        """

        # Compute Expected Loss
        el = exposure[:, :-1] * self.el[:, :-1]
        el = tf.reduce_sum(el, axis=-1, keepdims=True) / tf.reduce_sum(exposure[:, :-1], axis=-1, keepdims=True)

        # Compute risk premium
        risk_premium = tf.exp(tf.maximum(el, self.cor_target) / self.cor_target - 1.0) - 1.0

        # Compute ROE >>>
        roe = tf.reduce_sum(exposure * (self.im - risk_premium - self.el - self.cta), axis=-1, keepdims=True) / self.t1

        return el, roe

    def risk(self, exposure):
        """
        Compute the diversification ratio of a portfolio.
        """

        # Reshape "el_std" to [1, N, 1]
        el_std = tf.expand_dims(self.el_std, axis=-1)

        # Compute the covariance matrix: σ * σ^T * R, shape = [1, N, N]
        covariance_matrix = tf.matmul(el_std, el_std, transpose_b=True) * self.el_corr

        # Compute weights, shape = [1, N, 1]
        weights = tf.expand_dims(exposure / tf.reduce_sum(exposure, axis=-1, keepdims=True), axis=-1)

        # Compute the portfolio variance: w^T * covariance_matrix * w
        portfolio_var = tf.matmul(tf.matmul(tf.transpose(weights, perm=[0,2,1]), covariance_matrix), weights)
        portfolio_std = tf.reduce_sum(tf.sqrt(portfolio_var), axis=-1)

        # Compute the weighted sum of individual asset standard deviations
        weighted_std = tf.reduce_sum(weights * el_std, axis=1)

        # Compute the diversification ratio
        diversification_ratio = portfolio_std / weighted_std

        return diversification_ratio

    def t1_ratio(self, exposure):
        """
        Compute Tier 1 Capital Ratio given segment allocations, risk weights and tier 1 capital.
        :param exposure: segment allocations, US$ bn.
        :param rw: Risk weights, unit scale,
        :param t1: Tier 1 capital, US$ bn.
        :return: Tier 1 Capital Ratio.
        """

        rwa = tf.reduce_sum(exposure * self.rw, axis=-1, keepdims=True)
        t1_ratio = tf.divide(self.t1, rwa + self.epsilon)
        return t1_ratio

    def tc_ratio(self, exposure):
        """
        Compute Total Capital Ratio given segment allocations, risk weights, tier 1 and tier 2 capital.
        :param exposure: segment allocations, US$ bn.
        :param rw: Risk weights, %.
        :param t1: Tier 1 capital, US$ bn.
        :param t2: Tier 2 capital, US$ bn.
        :return: Total Capital Ratio.
        """

        rwa = tf.reduce_sum(exposure * self.rw, axis=-1, keepdims=True)
        tc = tf.add(self.t1, self.t2)
        tcr = tf.divide(tc, rwa + self.epsilon)
        return tcr

    def leverage(self, exposure):
        """
        Compute Leverage Ratio given segment allocations and  tier 1 capital.
        :param exposure: segment allocations, US$ bn.
        :param t1: Tier 1 capital, US$ bn.
        :return: Leverage Ratio.
        """

        exposure = tf.reduce_sum(exposure, axis=-1, keepdims=True)
        leverage = tf.divide(self.t1, exposure)
        return leverage

    def is_feasible(self, metric, threshold, mode: str):
        """
        Check if asset allocation is feasible.
        """

        threshold = tf.expand_dims(tf.constant(threshold, dtype=tf.float32), axis=0)

        if mode == 'min':
            indicator = tf.cast(metric >= threshold, dtype=tf.int32)
        elif mode == 'max':
            indicator = tf.cast(metric <= threshold, dtype=tf.int32)
        else:
            threshold = tf.gather(threshold, self.soft_limit, axis=1)
            metric = tf.gather(metric, self.soft_limit, axis=1)
            x1 = tf.cast(metric <= self.epsilon, dtype=tf.int32)
            x2 = tf.cast(metric >= threshold, dtype=tf.int32)
            indicator = tf.maximum(x1, x2)

        return tf.reduce_prod(indicator, axis=-1, keepdims=True)


class ModelLoss(losses.Loss):
    """
    Aggregate loss function: L = Z * [I(C1) * I(C2) * ... * I(Cn)], where:
        * Z - objective function: ROE to diversification rate,
        * C - constraints: expected loss, ROE, capital adequacy ratios, portfolio structure, asset class lower/upper limits.
        * I - continuous indicator function (see "indicator" method for details),
        * n - number of asset classes.
    """
    def __init__(self, objective: str, mode: str, drop: list, beta: float,
                 t1_min: float, tc_min: float, ccb_min: float, leverage_min: float,
                 roe_min: float, el_max: float,
                 exposure_min: list, soft_limit: list, exposure_max: list, concentration_min: float,
                 name='loss', **kwargs):
        super(ModelLoss, self).__init__(name=name)
        self.objective = objective
        self.mode = tf.constant(mode)
        self.soft_limit = tf.where(soft_limit)[:, 0]
        self.portfolio_metrics = PortfolioMetrics(**parameters)
        self.indicators = {
            'roe': {'mode': 'min', 'threshold': roe_min},
            't1_ratio': {'mode': 'min', 'threshold': t1_min},
            'tc_ratio': {'mode': 'min', 'threshold': tc_min},
            'ccb_ratio': {'mode': 'min', 'threshold': ccb_min},
            'leverage': {'mode': 'min', 'threshold': leverage_min},
            'el': {'mode': 'max', 'threshold': el_max},
            'exposure_min': {'mode': 'soft', 'threshold': exposure_min},
            'exposure_max': {'mode': 'max', 'threshold': exposure_max},
            'concentration': {'mode': 'min', 'threshold': concentration_min}}
        for x in drop:
            self.indicators.pop(x)
        self.beta = tf.Variable(initial_value=beta, trainable=False)
        self.epsilon = 1e-07
        self.reduction = None

    def call(self, y_true, y_pred):

        # Compute portfolio metrics
        metrics = self.portfolio_metrics.call(y_pred)

        # Compute sigmoid activations ("indicators") for constraints
        indicators = [self.indicator(metric=metrics[x], **self.indicators[x]) for x in self.indicators.keys()]
        indicators = tf.concat(indicators, axis=-1)
        indicators = tf.reduce_prod(indicators, axis=-1, keepdims=True)

        # Compute objective and loss
        objective = tf.cond(
            pred=tf.equal(self.mode, tf.constant('max')),
            true_fn=lambda: -metrics[self.objective],
            false_fn=lambda: tf.math.log(metrics[self.objective]))
        loss = tf.squeeze(objective * indicators, axis=-1)

        return loss

    def indicator(self, metric, threshold, mode: str):
        """
        Sigmoid function to penalize loss for breaking the constraints while granting zero benefit for the extra
        capacity (extra capital, etc.) and keeping loss differentiable.
        Beta is gradually increased via callbacks to ensure numerical stability and handle vanishing grads.
        :param mode: ['min', 'max', 'soft'], where 'min' = "greater or equal", 'max' = "less equal" and
        "soft" = "zero or greater or equal" (corporate and consumer business segment exposure) constraints.
        """

        # Convert threshold to tf.Tensor
        threshold = tf.expand_dims(tf.constant(threshold, dtype=tf.float32), axis=0)

        # Compute metric/threshold ratio (if threshold = min) or threshold/metric (threshold = max)
        # Apply reduction for vector inputs
        if mode in ['min', 'max']:
            ratio = metric / threshold if mode == 'min' else threshold / (metric + self.epsilon)
            ratio = ratio - 1.0
            indicator = tf.nn.sigmoid(ratio * self.beta.value())
            indicator = tf.reduce_prod(indicator, axis=-1, keepdims=True)

        # Soft limits of form "greater than or zero"
        # Apply reduction for vector inputs
        else:
            threshold = tf.gather(threshold, self.soft_limit, axis=1)
            metric = tf.gather(metric, self.soft_limit, axis=1)

            # 1st term sets indicator to 1 for values in range [0, epsilon]
            # 2nd term sets indicator to 1 for values above threshold
            ratio1 = metric / self.epsilon
            x1 = 1 - tf.nn.sigmoid(ratio1 * self.beta.value())
            ratio2 = metric / threshold - 1.0
            x2 = tf.nn.sigmoid(ratio2 * self.beta.value())
            indicator = x1 + x2
            indicator = tf.reduce_prod(indicator, axis=-1, keepdims=True)  # Apply reduction for vector inputs

        return indicator


class ModelMetrics(keras.metrics.Metric):
    """
    Metrics: compute and report metrics during training and inference.
    """
    def __init__(self, objective: str, mode: str, drop: list, name='loss', **kwargs):
        super(ModelMetrics, self).__init__(name=name)
        self.objective = objective
        self.mode = tf.constant(mode)
        self.portfolio_metrics = PortfolioMetrics(**parameters)
        metrics = ['mrtg', 'cstn', 'corp', 'cnsr', 'mmkt',
                   'raroc', 'roe', 'risk', 'el',
                   't1_ratio', 'tc_ratio', 'ccb_ratio', 'leverage', 'concentration']
        metrics = [x for x in metrics if x not in drop]
        self.metrics = {key: tf.Variable(initial_value=0.0, trainable=False, shape=()) for key in metrics}

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metrics with the best feasible solution.
        """

        # Helper functions >>>

        def update_metrics():
            metrics = self.portfolio_metrics.call(y_pred)
            sample = tf.cond(
                pred=tf.equal(self.mode, tf.constant('max')),
                true_fn=lambda: tf.argmax(metrics[self.objective], axis=0)[0],
                false_fn=lambda: tf.argmin(metrics[self.objective], axis=0)[0])
            for key in self.metrics.keys():
                self.metrics[key].assign(metrics[key][sample, 0])

        def no_op():
            return

        # Gather feasible solutions -> Update state with the "best" feasible solution
        metrics = self.portfolio_metrics.call(y_pred)
        mask = tf.where(metrics['feasibility'])[:, 0]
        y_pred = tf.gather(y_pred, mask, axis=0)
        tf.cond(tf.size(y_pred) > 0, update_metrics, no_op)

    def result(self, *args, **kwargs):
        return self.metrics


class BetaCallback(callbacks.Callback):
    """
    On train step end: increase "beta" (scale parameter of sigmoid indicator function) by "step".
    On epoch end: print exposure per asset class, US$ bn.
    """
    def __init__(self):
        super(BetaCallback, self).__init__()
        self.step = tf.Variable(initial_value=0.0)

    def on_train_batch_end(self, batch, logs=None):
        self.step.assign_add(1.0)
        self.model.loss.beta.assign(tf.sqrt(self.step))


class ExposureConstraint(constraints.Constraint):
    """
    Set "hard" constraints for exposure per asset class.
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)


def optimization(objective: str, mode: str, learning_rate: float, decay_rate: float, decay_steps: float,
                 batch_size, **kwargs):
    """
    Optimize allocations via custom gradient-based optimization.
    Convert constraints to ratios and apply scaled sigmoid: C = sigmoid((ratio-1) * beta).
    Gradual increase of "beta" (sigmoid scale parameter) to reduce threshold allowance while keeping stable grads.
    See Loss docs for complex constrains of type "Zero or greater than".
    Use objective function of form Z = F * C1 * C2 * ... Cn.
    Use single Dense layer with number of units = number of classes, where weights = allocations.
    Weights constraints: non-negativity and  "hard" min/max constraints.
    """

    # Dataset of input vectors
    ds = (tf.ones(shape=(1,), dtype=tf.float32), tf.ones(shape=(5,), dtype=tf.float32))
    ds = tf.data.Dataset.from_tensors(ds)
    ds = ds.repeat().batch(batch_size=batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build and compile the model
    model = AllocationModel()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_rate=decay_rate, decay_steps=decay_steps)),
        loss=ModelLoss(**parameters),
        metrics=[ModelMetrics(**parameters)])
    x = tf.constant(shape=(batch_size, 1), value=1.0)
    model(x)
    print(model.summary())

    # Train the model
    model.fit(ds, epochs=10000, steps_per_epoch=10000, callbacks=[
        callbacks.EarlyStopping(
            monitor=objective, mode=mode,
            patience=10, restore_best_weights=True, min_delta=0.01, start_from_epoch=10),
        BetaCallback()])

    # Print optimal allocations
    y_pred = model(x)
    metrics = ModelMetrics(**parameters)
    metrics.update_state(y_pred, y_pred)
    metrics = metrics.result()
    for key in metrics:
        print(key, metrics[key].numpy())
    return model

# EXECUTION >>>

optimization(**parameters)
