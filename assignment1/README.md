# Assigment 1

Extend the [`dg1` demo][1] in one or several of the following ways:

-   Implement support for arbitrary polynomial degrees
    -   Gaussian quadrature for integrals
    -   Use Gauss-Lobatto nodes instead of equidistant nodes
    -   Legendre polynomials in Vandermonde matrix (not monomials)
    -   Test for very high `p` (10-15), measure convergence order
-   Replace the RK4 explicit time-stepper by an implicit scheme,
    e.g. a 3rd order DIRK scheme
    -   Need to form and solve the full matrix
    -   Particularly useful for high `p`, to avoid restrictive CFL limits
-   Implement support for nonlinear (arbitrary?) equations (that is,
    flux function `f(u)`)
    -   Cannot do the linear matrix approach, need to recompute integrals
        on-the-fly
    -   Might need to code inner loop in a compiled language or vectorize
        for speed
    -   Test on Burgers' equation, see [228B notes][3] for Riemann solver (but
        don't let the shock form, it will blow up! more on this later)
-   Extend to 2-D, for a rectangular domain with square (Cartesian) elements
    -   Need to develop some better data structures, plotting utilities, etc
    -   Can in principle use the same approach as before with precomputed
        matrices
    -   Consider either full DG (with 2-D integrals and edge integrals) or
        the simpler Line-DG method (just adding up 1-D stencils and point-wise
        numerical fluxes)
    -   Test on "spinning Gaussian", confirm high order of convergence
-   Extend to 2-D, for unstructured meshes of triangles
    -   Again new data structures and plotting needed (see [`meshutils`][2]
        for some existing tools)
    -   Could still use matrix approach, but need a large sparse matrix
        since all elements are different. Or re-assemble on-the-fly.
    -   If needed: Simplify by only supporting `p = 1`

---------------------

**Due date**: February 11

[1]: https://nbviewer.jupyter.org/url/persson.berkeley.edu/228B/notebooks/dg1.ipynb
[2]: https://nbviewer.jupyter.org/url/persson.berkeley.edu/228B/notebooks/meshutils.ipynb
[3]: https://github.com/dhermes/berkeley-m273-s2016/tree/master/228B_notes
