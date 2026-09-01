# CI and Jenkins container debt

This record intentionally describes deferred work. The changes below affect
shared Jenkins agents, NFS image transport or job definitions and require team
agreement before implementation.

## Remove the Jenkins compatibility launcher

The root ``run-docker.sh`` is no longer a supported user interface. It remains
only because these Jenkins definitions still invoke its historical command
syntax:

- ``ci/Jenkinsfile``
- ``ci/Jenkinsfile_CI``
- ``ci/Jenkinsfile_Brevitas``
- ``docker/jenkins/Jenkinsfile_HW``

Migrate active jobs to explicit ``docker/build`` and ``docker/run`` commands,
including an explicit grant tier and runtime set, then delete the root shim.
Do not perform a blind command-name substitution: the old launcher inferred
defaults that the current interface requires Jenkins to state deliberately.

## Digest verification after image transport

`ci/scripts/build-images.sh` records an image digest, but
`ci/scripts/load-shared-image.sh` currently validates only the requested tag.
After `docker load`, the loader should compare the loaded image ID with the
recorded digest and fail before tests run if they differ.

Questions for the team:

- Should shards run by image ID/digest rather than by the reattached tag?
- Is the provenance JSON the transport contract, or should digest/tag files
  remain separate compatibility artifacts?
- If a registry becomes available later, should NFS archives be replaced by
  digest-addressed pulls? A registry is not assumed by the current design.

## Docker cache cleanup

The main Jenkins pipeline runs `docker system prune -a -f` before building. This
is daemon-wide and assumes a single-tenant, non-concurrent builder. Prefer
builder-scoped BuildKit garbage collection or a dedicated builder instance.

Before changing it, confirm agent tenancy, concurrency and available disk
budgets with the CI owners.

## Legacy pipeline definitions

Several secondary pipeline files predate the current container model:

- `ci/Jenkinsfile_CI` sets legacy image-tag variables.
- `ci/Jenkinsfile_Brevitas` edits dependency pins in `fetch-repos.sh`, although
  pins now live in `deps.env`.
- `docker/jenkins/Jenkinsfile_HW` duplicates cleanup and report behavior from
  the main pipeline.

The team should identify which jobs still load each file. Active behavior should
move to shared helpers; inactive definitions should be archived or removed so
they no longer appear to be supported entry points.

## Provenance completeness

The current provenance record is strongest when `deps.env` contains immutable
commit IDs. Branch or tag overrides are recorded as unresolved. Decide whether
CI should reject unresolved dependency refs or resolve them through an approved
network operation before publishing an image.
