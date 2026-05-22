# Jenkins

## Pipeline model
Jenkins Pipeline is usually defined in a `Jenkinsfile`. The controller stores jobs and schedules work; agents execute build steps. Stages organize the flow, and post actions handle cleanup, notifications and artifact collection.

## Troubleshooting
Common failures are missing agent labels, broken credentials binding, workspace leftovers, plugin incompatibility and environment differences between controller and agent.
