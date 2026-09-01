# FINN image identity

FINN images are periodically rebuilt environments, not bit-for-bit reproducible
artifacts identified by their human-readable tag.

The supported workflow builds images locally. A remote OCI registry is not
assumed: Docker runs the local image, sbx imports it into its own image store,
and the Apptainer backend converts it into a local SIF.

The tag records source provenance and image flavor: the FINN revision, selected
runtime packages, and whether the sbx contract is present. It is convenient for
selection, but it is mutable. A fresh build can pick up newer packages from the
configured apt and Python indexes while retaining the same tag.

The immutable identity of one built image is its content digest. When recording,
promoting or comparing an environment, use:

1. the image digest;
2. the FINN source commit;
3. the resolved dependency commits from `deps.env`; and
4. the selected runtime set.

An SBOM describes the OS and Python packages contained in that digest. Local or
CI builds should retain an SPDX or CycloneDX SBOM when the build tooling supports
it. A tag or SBOM without the digest is not sufficient to prove which
environment ran.

For local development, rebuilding may replace the image behind the same tag.
This is expected. Use `docker image inspect` when an exact local image ID matters.
