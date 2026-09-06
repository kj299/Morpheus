# Search-head validation

The app's newest half has never met a search head. This package makes that verdict cost minutes and produce a
diff against a written expectation, rather than a judgement call at the end of an afternoon of setup.

## Running it

```bash
cd examples/splunk_lineage_app/validate
docker compose up -d
# wait for the health check to go green, then index the sample events
docker exec morpheus-lineage-validate bash -lc '
  for f in /sample_events/*.jsonlines; do
    st=$(basename "$f" .jsonlines | tr "_" ":")
    case "$st" in
      morpheus:score:*|morpheus:edge) idx=behavior_events ;;
      binding:*)                      idx=behavior_bindings ;;
      *)                              idx=behavior_events ;;
    esac
    [ "$st" = "morpheus:edge" ] && idx=behavior_lineage
    /opt/splunk/bin/splunk add oneshot "$f" -index "$idx" -sourcetype "$st" -auth admin:"$SPLUNK_PASSWORD"
  done'
```

Then run each saved search and compare against [`expected_results.json`](./expected_results.json).

## The two things worth checking before any search

These are the failures this app has actually had, and both are invisible in a search result:

1. **`_time` must not be index time.** Run
   `index=behavior_events sourcetype=morpheus:score:l2 | eval drift = _time - _indextime | stats max(drift)`.
   The events are historical, so the drift should be large and negative. A drift near zero means `TIME_PREFIX`
   matched nothing and every windowed rule has quietly become a rule about when the data was loaded.
2. **`binding_table` must be present on bucketed rows.** Run
   `index=behavior_bindings sourcetype=binding:bucketed | stats count BY binding_table`. One row, `dhcp_lease`,
   80 events. An empty result means the refresh search selects on a field the producer stopped writing, and the
   lookup silently stops being refreshed.

## What each search should return

Full detail with reasons in [`expected_results.json`](./expected_results.json). Summary, including the ones that
should return nothing:

| Search | Rows | Note |
|---|---|---|
| R-D-L2-001, MAC count on an access port | **0** | Correct. Joins `port_designations`, which ships header-only; 11 candidate rows are waiting behind it. |
| R-D-L2-003, ARP anomaly | **1** | 24 contested observations aggregate to one notable on `10.0.0.1`. |
| R-D-L2-004, MAC in two places | **2** | A conflict at zero gap and a displacement at two seconds. The roaming device, displaced a full poll cadence later, is deliberately outside the threshold. |
| R-D-L2-005, authorization without authentication | **2** | One bypass on a quiet port, one that arrived while a legitimate exchange was open. |
| R-C-002, TLS before beaconing | **0** | Correct. Needs layer 4 and layer 7 telemetry; neither class exists. |
| Behavior summary, per-layer scores | **0** | Correct. Reads `max_abs_z`, which no shipped stage emits. Cannot receive a row until a third layer ships. |
| Chain assembly, cross-layer risk | **0** | Correct, and for the same reason: only two layers produce events. |
| Binding lookup, L2/L3 refresh | **80** | Written into the `binding_l2_l3` collection. |
| Binding lookup, L1 refresh | **0** | Correct. Nothing produces `binding:l1`; it needs a layer 1 inventory feed. |
| Binding lookup, L2/L3 expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Binding health, unresolved rate | **1** | An operational metric; the value matters, not whether it fired. |

**Six of the eleven should return nothing.** That is the point of writing them down. An empty result is this app's
characteristic failure, and without a list saying which emptiness is correct, a deployment cannot tell a rule that
is working from a rule that is broken.

## Where the sample events come from

[`make_sample_events.py`](./make_sample_events.py) runs the same `run_pipeline` the determinism tests call and
puts the output through the same `SiemWireStage` a deployment would put before its sink. They are checked in so
that a change to what a SIEM would receive shows up in a pull request rather than only on a search head.

Regenerate after changing the corpus or a stage, and review the diff:

```bash
python examples/splunk_lineage_app/validate/make_sample_events.py
```

## What this does not establish

That a particular Splunk version parses the app the way `props.conf` says. This runs one version, in one
container, with everything on one instance. A real estate parses on indexers or heavy forwarders, and the
`KV_MODE` and acceleration choices in `props.conf` are estate-wide decisions this package does not make.
