# Third-party notice: Xeryon Python Library v1.88

This package contains the `Xeryon.py` source distributed by Xeryon for its
Python library, version v1.88, with one local rotary `setSpeed` encoding fix
(`int(round(speed * 100))` instead of `int(speed) * 100` so SSPD uses 0.01 deg/s):

<https://xeryon.com/software/xeryon-python-library/>

The source was retrieved on 2026-09-10 from the vendor's website-distributed
ZIP. The SHA-256 of that ZIP is:

```
7f3c8a373e66b2fd2cb915035bc2c29f860b59ff76ec4ea1531824b3ca7b8c57
```

The ZIP digest above is the unmodified upstream archive. The vendored
`Xeryon.py` SHA-256 after the local `setSpeed` patch is:

```
bff3338ecff97c2cb01c18a3d07dafdd1b86a19f2e5ce2a8ac8c21a9825bce77
```

The narrow typed boundary in `xeryon_vendor.facade` is project-authored. Use of
this vendored source is authorized for this private repository only. Public
redistribution or publication of the vendor source requires separate
confirmation of Xeryon's licensing terms.

The vendor library uses `pyserial`; this workspace constrains it to
`pyserial>=3.5,<4`.
