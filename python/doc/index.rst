.. image:: _static/logo.svg
  :alt: CASM logo
  :width: 600

libcasm-mapping
===============

The libcasm-mapping package is the CASM structure mapping module. This includes:

- Methods for searching for low-cost lattice, atom, and structure mappings, taking into account symmetry, based on the approach described in the paper `Thomas, Natarajan, and Van der Ven, npj Computational Materials, 7 (2021), 164 <https://doi.org/10.1038/s41524-021-00627-0>`_.
- Methods for generating interpolated structures based on mapping results
- Methods for generating symmetrically equivalent mappings
- Data structures and methods for creating custom mapping searches


About CASM
==========

The libcasm-global package is part of the CASM_ open source software package, which is designed to perform first-principles statistical mechanical studies of multi-component crystalline solids.

CASM is developed by the Van der Ven group, originally at the University of Michigan and currently at the University of California Santa Barbara.

For more information, see the `CASM homepage <CASM_>`_.


License
=======

GNU Lesser General Public License (LGPL). Please see the LICENSE file available on GitHub_.


Documentation
=============

.. toctree::
    :maxdepth: 2

    Installation <installation>
    Usage <usage>
    Reference <reference/libcasm/index>
    Bibliography <bibliography>
    About libcasm-mapping <about>

libcasm-mapping is available on GitHub_.

.. _CASM: https://prisms-center.github.io/CASMcode_docs/
.. _GitHub: https://github.com/prisms-center/CASMcode_mapping
