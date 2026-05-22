# Terraform

## State and desired configuration
Terraform compares configuration with state and provider data, then creates a plan. State maps Terraform resources to real infrastructure IDs, so losing or corrupting state can be more serious than losing generated plan output.

## Modules and providers
Providers implement API calls to target systems. Modules package reusable resources. Keep variables explicit, outputs minimal and state remote with locking for team usage.
