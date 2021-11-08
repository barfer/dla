.. dla_mstm documentation master file, created by
   sphinx-quickstart on Tue Oct 19 17:44:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dla_mstm's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Download https://eng.auburn.edu/users/dmckwski/scatcodes/
and GFortran https://gcc.gnu.org/wiki/GFortran#news
gfortran mpidefs-serial.f90 mstm-intrinsics-v3.0.f90 mstm-modules-v3.0.f90 mstm-main-v3.0.f90 -O2 -o mstm.out

|

DLA module
----------
.. automodule:: dla
      :members:

|

Collision module
----------------
.. automodule:: collision
      :members:

|

Aggregate module
----------------
.. automodule:: aggregate
      :members:

|

Spherical module
----------------
.. automodule:: spherical
      :members:

|

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
