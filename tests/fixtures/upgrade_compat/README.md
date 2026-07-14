# upgrade_compat fixtures

Vendored, committed copies of the templates `main`'s `lobes init` produced —
the "previous version" scaffold that `tests/test_upgrade_compat.py` proves
the CURRENT code keeps operating without a re-init.

Source commit: `main` @ `7fa624d1c891c2bc1c5934947d0d9379462605e0`
(`cd2dbb7 fix(changelog): one '### Fixed' per release (markdownlint MD024)`).

Vendored verbatim via:

```sh
git show main:lobes/templates/docker-compose.yml       > single/docker-compose.yml
git show main:lobes/templates/env.example               > single/env.example
git show main:lobes/templates/fleet/docker-compose.yml  > fleet/docker-compose.yml
git show main:lobes/templates/fleet/env.example          > fleet/env.example
```

`lobes init` copies a template's content byte-for-byte into the deployment dir
(`env.example` -> `.env`, `fleet/docker-compose.yml` -> `docker-compose.yml`),
so these files ARE what an old scaffold's `docker-compose.yml`/`.env` contain
— no further materialisation step needed beyond a plain copy + rename.

Do not hand-edit these files; re-vendor from a later `main` commit if the
fixture ever needs to move forward (the invariant under test is that the
CURRENT code still operates whatever these currently say).
