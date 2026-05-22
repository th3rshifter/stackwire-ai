# Prometheus

## Metrics and labels
In PromQL, a metric name and labels are different. A label alone is not a time series. Use a metric selector like `http_requests_total{job="api"}` and aggregate with `sum by (...)`, `avg by (...)` or similar operators.

## Alerts and SLO
Burn rate normally means error-budget consumption rate for an SLO. For CPU or memory alerts, avoid calling it burn rate unless the alert is actually tied to an SLO.

## Checks
Debug missing data by checking scrape targets, labels, metric name, time range, relabeling rules and exporter logs.
