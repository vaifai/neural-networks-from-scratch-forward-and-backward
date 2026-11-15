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

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

