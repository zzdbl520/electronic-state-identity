# PySCF smoke-test reference

The production code paths were exercised with PySCF 2.14.0 at
`lambda = 2.62425 bohr`, using the locked cc-pVTZ/CAS(6,6) model. The test
compared fresh-RHF and common-A1/B2-SA2 orbital initializations for both SA3
and SA4 in the A1 and B2 blocks.

## Pairwise diagnostics

| Irrep | Protocol | absolute delta E_average / Eh | maximum principal angle / rad | state leakage | Procrustes 1-RDM distance | NOON RMS |
|---|---:|---:|---:|---:|---:|---:|
| A1 | SA3 | 1.0386359902e-2 | 8.1099253180e-1 | 3.2787582264e-1 | 9.3725237661e-1 | 3.6973580199e-1 |
| A1 | SA4 | 9.5750562116e-3 | 8.8833263430e-1 | 2.4280052920e-1 | 7.2498445647e-1 | 2.5504659710e-1 |
| B2 | SA3 | 7.6383344094e-13 | 6.4705021781e-6 | 2.1813995055e-11 | 8.6324039084e-7 | 4.8434582169e-8 |
| B2 | SA4 | 3.1796787425e-13 | 1.3098027451e-5 | 1.1889600415e-11 | 3.6174503426e-7 | 5.4856933591e-8 |

The A1 calculations retain two strongly distinct stationary solutions, whereas
the B2 calculations return to the same solution to numerical precision. A1 is
therefore the provisional switching block. This is not yet the final B0 result:
the complete five-point left/right continuation test is required to establish
branch reproducibility and the spatial support of the switch.
