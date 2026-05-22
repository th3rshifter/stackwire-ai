# GitLab CI

## Pipeline model
`.gitlab-ci.yml` defines stages and jobs. A runner executes jobs in a chosen executor, such as shell, Docker or Kubernetes. Artifacts pass files between jobs; cache speeds repeated dependency downloads.

## Practical checks
When a job fails, check runner availability, executor image, variables, working directory, artifact paths and cache key. Keep deploy credentials in protected variables and avoid printing secrets.
