MPS to circuit
==============

Toolbox for converting matrix product states to Qiskit quantum circuits.

Usage with Quimb
----------------

Assuming you have a `Quimb <https://github.com/jcmgray/quimb>`__
``MatrixProductState`` object, you can extract the individual tensors as
three-dimensional NumPy arrays using the snippet below.

.. code:: python

   mps_arrays = mps.arrays

You can now convert the list of tensors to a Qiskit ``QuantumCircuit``
using the ``mps_to_circuit`` interface function, specifying the shape as
:math:`(v_L, v_R, p)` (this is also the default argument for ``shape``).

For example, using the “exact” method
[`Shoen2006 <https://arxiv.org/abs/quant-ph/0612101>`__]:

.. code:: python

   qc = mps_to_circuit(mps_arrays, method="exact", shape="lrp")

Or using the “approximate” method
[`Ran2019 <https://arxiv.org/abs/1908.07958>`__]:

.. code:: python

   qc = mps_to_circuit(mps_arrays, method="approximate", shape="lrp", num_layers=3)

Usage with TenPy
----------------

Assuming you have a `TenPy <https://github.com/tenpy/tenpy>`__ ``MPS``
object, you can extract the individual tensors as three-dimensional
NumPy arrays using the snippet below.

.. code:: python

   mps_arrays = [mps.get_B(i).to_ndarray() for i in range(L)]

You can now convert the list of tensors to a Qiskit ``QuantumCircuit``
using the ``mps_to_circuit`` interface function, specifying the shape as
:math:`(v_L, p, v_R)`.

For example, using the “exact” method
[`Shoen2006 <https://arxiv.org/abs/quant-ph/0612101>`__]:

.. code:: python

   qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")

Or using the “approximate” method
[`Ran2019 <https://arxiv.org/abs/1908.07958>`__]:

.. code:: python

   qc = mps_to_circuit(mps_arrays, method="approximate", shape="lpr", num_layers=3)

Development
-----------

Installation
~~~~~~~~~~~~

This package and all dependencies can be installed using
`uv <https://github.com/astral-sh/uv>`__.

.. code:: sh

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12
   uv venv --python 3.12
   uv python pin 3.12
   uv pip install -e .

Testing
~~~~~~~

One can test the code using
`pytest <https://github.com/pytest-dev/pytest>`__ .

.. code:: sh

   uv run pytest

Linting
~~~~~~~

One can lint the code using the `Ruff
Linter <https://docs.astral.sh/ruff/linter/>`__.

.. code:: sh

   uv run ruff check

Formatting
~~~~~~~~~~

One can format the code using the `Ruff
Formatter <https://github.com/astral-sh/ruff/formatter/>`__.

.. code:: sh

   uv run ruff format

Sorting imports
~~~~~~~~~~~~~~~

In order to both sort imports and format, call the Ruff linter and then
the formatter:

.. code:: sh

   uv run ruff check --select I --fix
   uv run ruff format

Code of conduct
~~~~~~~~~~~~~~~

All members of this project agree to adhere to the Qiskit Code of
Conduct listed
`here <https://github.com/Qiskit/qiskit/blob/master/CODE_OF_CONDUCT.md>`__.

License
-------

`Apache-2.0 license <LICENSE.txt>`__

Acknowledgements
----------------

This work was supported by the Hartree National Centre for Digital
Innovation, a collaboration between the Science and Technology
Facilities Council and IBM.
