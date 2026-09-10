# Third-party notice: Xeryon Python Library v1.88

This package contains the unmodified `Xeryon.py` source distributed by Xeryon
for its Python library, version v1.88:

<https://xeryon.com/software/xeryon-python-library/>

The source was retrieved on 2026-09-10 from the vendor's website-distributed
ZIP. The SHA-256 of that ZIP is:

```
7f3c8a373e66b2fd2cb915035bc2c29f860b59ff76ec4ea1531824b3ca7b8c57
```

The source file is preserved byte-for-byte; the narrow typed boundary in
`xeryon_vendor.facade` is project-authored and does not modify the vendor
source. Use of this vendored source is authorized for this private repository
only. Public redistribution or publication of the vendor source requires
separate confirmation of Xeryon's licensing terms.

The vendor library uses `pyserial`; this workspace constrains it to
`pyserial>=3.5,<4`.
