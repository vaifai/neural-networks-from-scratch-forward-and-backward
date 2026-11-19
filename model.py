"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x,dtype=float)
    x_copy = np.array(x,dtype=float,copy=True)
    for idx in np.ndindex(x.shape):
        orig = x_copy[idx]
        x_copy[idx] = orig + eps
        f_plus = f(x_copy)
        x_copy[idx] = orig - eps
        f_minus = f(x_copy)
        grad[idx] = (f_plus - f_minus)/(2*eps)
        x_copy[idx] = orig
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)
    diff = analytic_grad - numeric_grad
    scale = np.maximum(np.abs(analytic_grad), np.abs(numeric_grad))
    scale = np.maximum(scale, tol)
    
    return np.max(np.abs(diff)/scale).item()

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W,b = weight_init_fn(in_dim, out_dim)

    def forward(x):
      cache = {}
      cache["W"] = W
      cache["b"] = b
      cache["x"] = x
      y = x @ W + b
      return y,cache
    
    def backward(dout, cache):
      dx = dout @ cache["W"].T
      dW = cache["x"].T @ dout
      db = np.sum(dout, axis=0)
      grads = {
        "W": dW,
        "b": db
      }
      return dx,grads
    
    return {
      "params" : {
        "W": W,
        "b": b,
      },
      "forward": forward,
      "backward": backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    def forward(x):
      cache = {}
      cache["x"] = x
      cache["activation"] = kind
      if kind == "relu":
        y = np.maximum(x,0)
        cache["y"] = y
        return y,cache
      elif kind == "tanh":
        y = np.tanh(x)
        cache["y"] = y
        return y,cache
      else:
        y = 1/(1 + np.exp(-x))
        cache["y"] = y
        return y,cache
      
    def backward(dout, cache):
      if cache["activation"] == "relu":
        dx = dout * (cache["x"] > 0)
        return dx, {}
      elif cache["activation"] == "tanh":
        dx = dout * (1 - cache["y"]**2)
        return dx, {}
      else:
        dx = dout * cache["y"] * (1 - cache["y"])
        return dx, {}
    return {
      "params": {},
      "forward": forward,
      "backward": backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    std = np.sqrt(2.0 / in_dim)
    W = np.random.randn(in_dim, out_dim) * std
    b = np.full((out_dim,), 1e-6)
    return W,b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    def loss_fn(logits, labels):
      N,C = logits.shape
      shifted = logits - np.max(logits, axis=1, keepdims=True)
      exp_z = np.exp(shifted)
      p = exp_z/np.sum(exp_z,axis=1, keepdims=True)
      loss = np.mean(-np.log(p[np.arange(N), labels]))
      y_onehot = np.eye(C)[labels]
      d_logits = (p - y_onehot)/N
      return loss,d_logits
    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    params = [layer["params"] for layer in layers]
    def forward(x):
      y, caches = x,[]
      for ind,layer in enumerate(layers):
        yy,cache = layer["forward"](y)
        y = yy
        caches.append(cache)
      return y,caches
    
    def backward(dout, caches):
      dx,grads_list = dout, [None] * len(layers)
      for ind, layer in reversed(list(enumerate(layers))):
        dx, grads = layer["backward"](dx, caches[ind])
        grads_list[ind] = grads
      return dx,grads_list
    return {
      "forward": forward,
      "backward": backward,
      "params": params
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    logits, caches = model["forward"](x)
    loss, d_logits = loss_fn(logits, y)
    dx, param_grads = model["backward"](d_logits, caches)
    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    def update(param, grad):
      if isinstance(param, np.ndarray):
        assert param.shape == grad.shape
        param-=lr*grad
      elif isinstance(param, dict):
        for key in param:
          update(param[key], grad[key])
      elif isinstance(param, list):
        for p, g in zip(param, grad):
          update(p,g)
    
    def step(grads):
      update(params, grads)
      
    
    return {
      "step": step
    }

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer["step"](param_grads)
    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    np.random.seed(seed)
    history = []
    for i in range(epochs):
      batches = int(np.ceil(len(x)/batch_size))
      total_loss = 0.0
      for j in range(0,len(x),batch_size):
        loss,param_grads = forward_backward(model, loss_fn,x[j:j+batch_size], y[j:j+batch_size])
        optimizer["step"](param_grads)
        total_loss+=loss
      history.append(total_loss/batches)
    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    x, y = generate_data(input_dim, num_classes, seed)
    np.random.seed(seed)

    def init_fn(i, o):
      return initialize_weights(i, o, scheme="he")
    
    layers = [
      make_dense(input_dim, 64, init_fn),
      make_activation("relu"),
      make_dense(64, 64, init_fn),
      make_activation("relu"),
      make_dense(64, num_classes, init_fn)
    ]

    model = make_sequential(layers)

    loss_fn = make_loss("cross_entropy")

    optimizer = make_optimizer(
      model["params"],
      lr=0.05,
      kind="sgd"
    )

    train(model,loss_fn,optimizer,x,y,epochs=300,batch_size=32,seed=seed)

    logits, _ = model["forward"](x)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y))

    metrics = {
      "accuracy": accuracy,
      "x": x,
      "y": y
    }

    return model, metrics

    
def generate_data(input_dim, num_classes, seed):
  rng = np.random.default_rng(seed)
  samples_per_class = 100

  xs = []
  ys = []

  for c in range(num_classes):
    for region in range(2):
      center = 2 * (region * num_classes + c)
      x_part = rng.normal(
        loc=0.0,
        scale=0.08,
        size=(samples_per_class // 2, input_dim)
        )
      x_part[:, 0] += center
      xs.append(x_part)
      ys.append(
        np.full(samples_per_class // 2, c, dtype=np.int64)
        )
  x = np.vstack(xs)
  y = np.concatenate(ys)

  x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
  perm = rng.permutation(len(x))
  return x[perm], y[perm]

# Step 13 - improve_generalization
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model

    Required behavior:
      val_accuracy > baseline_val_accuracy
      predictions == argmax(model.forward(x_val), axis=1)
      val_accuracy == mean(predictions == y_val)
      predictions are non-constant (not a trivial single-class predictor)
    """
    # TODO: your approach here

    def create_optimizer(params, lr=0.05, weight_decay=0.0):
      def update(param, grad, key=None):
        if isinstance(param, np.ndarray):
          if key == "W":
            param -= lr * (grad + weight_decay * param)
          else:
            param-=lr*grad
        elif isinstance(param, dict):
          for k in param:
            update(param[k],grad[k],k)
        elif isinstance(param, (list, tuple)):
          for p,g in zip(param, grad):
            update(p,g)
      
      def step(grads):
        update(params, grads)
      
      return {"step": step}

    
    def evaluate(model):
      logits, _ = model["forward"](x_val)
      predictions = np.argmax(logits, axis=1)
      accuracy = float(np.mean(predictions == y_val))
      return accuracy, predictions

    def save_params(params):
      if isinstance(params, np.ndarray):
        return params.copy()
      
      if isinstance(params, dict):
        return {
          k:save_params(v) for k,v in params.items()
        }
      if isinstance(params, (list, tuple)):
        return [save_params(p) for p in params]
      
      raise TypeError("Unsupported parameter type")

    def restore_params(params, saved):
      if isinstance(params, np.ndarray):
        params[...] = saved
      elif isinstance(params, dict):
        for k in params:
          restore_params(params[k], saved[k])
      elif isinstance(params, (list, tuple)):
        for p,s in zip(params,saved):
          restore_params(p,s)
      else:
        raise TypeError("Unsupported parameter type")

    np.random.seed(seed)
    baseline = baseline_model_fn()

    improved = baseline_model_fn()
    initial_params = save_params(baseline["params"])
    restore_params(improved["params"], initial_params)

    loss_fn = make_loss()
    lr = 0.05
    max_epochs = 70
    batch_size = 32

    baseline_optimiser = make_optimizer(baseline["params"], lr=lr, kind = "sgd")
    improved_optimiser = create_optimizer(improved["params"], lr=lr, weight_decay=0.05)

    train(
      baseline,
      loss_fn,
      baseline_optimiser,
      x_train,
      y_train,
      epochs=70,
      batch_size=batch_size,
      seed=seed
    )

    baseline_accuracy, _ = evaluate(baseline)

    best_accuracy, _ = evaluate(improved)
    best_params = save_params(improved["params"])

    best_epoch = 0
    patience = 40

    for epoch in range(150):
      train(
        improved,
        loss_fn,
        improved_optimiser,
        x_train,
        y_train,
        epochs=1,
        batch_size=batch_size,
        seed=seed + epoch
      )

      val_accuracy, _ = evaluate(improved)
      if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = save_params(improved["params"])
        best_epoch = epoch + 1

      if (epoch + 1) - best_epoch >= patience:
        break

    restore_params(improved["params"], best_params)
    val_accuracy, predictions = evaluate(improved)

    return {
      "val_accuracy": val_accuracy,
      "baseline_val_accuracy": baseline_accuracy,
      "predictions": predictions,
      "model": improved
    }

