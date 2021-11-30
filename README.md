# RAMA: Rapid algorithm for multicut problem
Solves multicut (correlation clustering) problems orders of magnitude faster than CPU based solvers without compromising solution quality on NVIDIA GPU. It also gives lower bound guarantees.

![animation](./misc/contraction_animation.gif)

## Requirements
We use `CUDA 11.2` and `GCC 10`. Other combinations might also work but not tested. `CMake` is required for compilation.

## Installation

### C++ solver:
```bash
mkdir build
cd build
cmake ..
make -j 4
```

### Python bindings:
We also provide python bindings using [pybind](https://github.com/pybind/pybind11). Simply run the following command:

```bash
python -m pip install git+https://github.com/pawelswoboda/RAMA.git
```

## Usage

### C++ solver:
We require multicut instance stored in a (.txt) file in the following format:
```
MULTICUT
i_1, j_1, cost_1
i_2, j_2, cost_2
...
i_n, j_n, cost_n
```
which corresponds to a graph with `N` edges. Where `i` and `j` should be vertex indices and `cost` is a floating point number. Positive costs implies that the nodes are similar and thus would prefer to be in same component and viceversa. Afterwards run:
```bash
./rama_text_input -f <PATH_TO_MULTICUT_INSTANCE>
```

### Python solver:
An example to compute multicut on a triangle graph:
```python
import rama_py
rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], rama_py.multicut_solver_options("PD")) 
```
The solver supports different modes which can be chosen by initializing multicut_solver_options by following:
- `"P"`: For running purely primal algorithm (best runtime).
- `"PD"`: This one offers good runtime vs quality tradeoff and is the default solver.
- `"PD+"`: For better quality primal algorithm (worse runtime). 
- `"D"`: For only computing the lower bound.
 
#### PyTorch support:
The above-mentioned Python solver takes input in CPU memory and then copies to GPU memory. In cases where this takes too much time we offer additional (optional) functionality in Python bindings which allow to directly use the GPU tensors and return the result in GPU memory. For this there are two options:

- ** Binding via pointers to GPU memory: **
Does not require compiling RAMA with PyTorch support (as done below). This option passes the GPU memory pointers to RAMA (the data is not modified). See 
'test\test_pytorch_pointers.py` for usage.

- ** Direct binding of Torch Tensors: **
To use this functionality ensure that PyTorch is built with the same CUDA version as the one used in this code and the ABI's match (see https://discuss.pytorch.org/t/undefined-symbol-when-import-lltm-cpp-extension/32627/7 for more info). Support for PyTorch can be enabled by:
```
WITH_TORCH=ON pip install setup.py
```
After this you should be able to run `test/test_pytorch.py` without any errors. To suppress solver command line output set `opts.verbose=False`.

### Parameters:
The default set of parameters are defined [here](include/multicut_solver_options.h) which correspond to algorithm `PD` from the paper. This algorithm offers best compute time versus solution quality trade-off.  Parameters for other variants are:

 - **Fast purely primal algorithm (P)**:
 This algorithm can be slightly worse than sequential CPU heuristics but is 30 to 50 times faster. 
	```bash
	./rama_text_input -f <PATH_TO_MULTICUT_INSTANCE> 0 0 0 0
	```
- **Better quality primal algorithm (PD+)** :
This algorithm can even be better than CPU solvers in terms of solution quality as it uses dual information. Still, it is 5 to 10 faster than best CPU solver.
	```bash
	./rama_text_input -f <PATH_TO_MULTICUT_INSTANCE> 5 10 5 10
	```
- **Dual algorithm (D)**:
Use this algorithm for only computing the lower bound. Our lower bounds are slightly better than [ICP](http://proceedings.mlr.press/v80/lange18a.html) and are computed up to 100 times faster.
	```bash
	./rama_text_input -f <PATH_TO_MULTICUT_INSTANCE> 5 10 0 0 5
	```
Run  `./rama_text_input --help` for details about the parameters. 
