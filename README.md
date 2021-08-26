# RAMA: Rapid algorithm for multicut problem
Solves multicut (correlation clustering) problems orders of magnitude faster than CPU based solvers without compromising solution quality using NVIDIA GPU. It also gives lower bound guarantees. 

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
which corresponds to a graph with `N` edges. Where `i` and `j` should be vertex indices and `cost` is a floating point number. Positive costs of an edge model preference of the corresponding nodes to be in the same component and viceversa. Afterwards just run:
```bash
./rama_text_input <PATH_TO_MULTICUT_INSTANCE>
```

### Python solver:
An example to compute multicut on a triangle graph:
```python
import rama_py
rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], rama_py.multicut_solver_options()) 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
