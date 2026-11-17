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

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

