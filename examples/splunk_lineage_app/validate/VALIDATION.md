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
| R-B-L3-001, fan-out expansion | **1** | One notable, for the scanner. The browser beside it reaches as many addresses and none of them inside the estate, which is the whole of the difference. |
| R-B-L3-002, beaconing | **1** | One notable, for the pair on a five-minute timer. The worker making plenty of flows to one file server at ragged intervals does not appear. |
| R-D-L3-003, reserved-range egress | **1** | One flow, to `240.0.0.1`. What the number really asserts is that the other 410 flows are classified correctly. |
| R-B-L3-004, TTL fingerprint shift | **1** | One notable: twelve flows from one source, every one short by exactly the single hop an interposed device costs. The steady host beside it never moves. |
| R-P-L3-005, fan-out trajectory | **0** | Correct. Reads `index=behavior_summary`, which the summary search populates as it runs and this package does not. The trajectory is in the data. |
| R-D-L4-002, SYN without completion | **1** | One notable: 60 destination ports in one bin, none of them answering. The workstation's handshakes complete, and the sweep touches one port across thirty hosts rather than sixty on one. |
| R-D-L4-003, RST ratio | **2** | Two notables from one predicate. 50 flows sit at the same ratio; one server refusing 20 clients is an outage, 30 servers refusing one client is enumeration, and they need different responses. |
| R-B-L4-005, transfer envelope breach | **1** | One notable: 60000 bytes against the triple's own envelope of 1200. The busy triple beside it moves more in total and never breaches, which is what a global threshold could not express. |
| R-D-L5-003, impossible travel | **2** | One principal in New York half an hour after her own London office login, and back in London ninety minutes later. Two rows is what one interloper produces. The eight-hour flight beside it, and the VPN user changing country twice a day, do not appear. |
| R-D-L5-004, multi-factor fatigue | **1** | Five denials in eight minutes and then an approval. The fumbled password beside it -- two failures and a success with the factor never challenged -- does not appear. |
| R-C-002, TLS before beaconing | **0** | Correct. Chains two detections' notables, and layer 6 has neither a producer nor a rule. It also reads `rule_id`, which detections write as they fire. |
| Behavior summary, per-layer scores | **890** | One row per five-minute bin, layer, entity and lineage over the 2586 scored events. It returned nothing until `EnvelopeStampStage` put `osi_layer` and `entity_key` on every record, and 320 until the estate pipeline rendered these events with the desk authentications beside the ports; `peak_z` is still null outside layer 5, because only `TC5ScoreStage` produces `max_abs_z`. |
| Chain assembly, cross-layer risk | **0** | Correct, and for a new reason. The threshold the search was built around is met: 15 of the 658 chains span three layers, holding a port's layer 1 samples, the layer 2 observations resolved onto it, and the authentications of the person sitting there. What stops it is the line after -- `total_risk >= 60 OR (layer_span >= 4 AND peak_z >= 4.0)` -- and `risk_score` is written into the index by the detection searches as they fire, not by a stage. This package indexes pipeline output alone, so every chain sums null. |
| Binding lookup, L2/L3 refresh | **0** | Correct, and it always was. The search selects `binding_table=dhcp_lease`; this corpus has no DHCP source, and the 80 bucketed rows it used to be credited with are a MAC table under a different name. |
| Binding lookup, L1 refresh | **5** | Five port intervals across four ports: three stable, two on the port whose optic is swapped. The lookup keys on port and switch with no bucket, so those two collapse to one row and the later optic wins -- it answers what is in a port now, not what was in it then. |
| Binding lookup, L1 history refresh | **1** | One row, and one row is the point. Only the port whose optic was replaced has a superseded interval; the other three are described for all time by the current-state row and cost the history nothing. |
| Binding lookup, L1 history expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Binding lookup, L2/L3 expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Binding health, unresolved rate | **1** | An operational metric; the value matters, not whether it fired. |
| R-P-L5-006, drift trajectory | **7** | Three principals, none of them behaviour, each explained in `expected_results.json`: two climb for six days because the reference scorer's baseline is frozen under cumulative features, one has a shallow run ended by the planted burst. Watchlist, never a page. |

**Seven of the twenty-four should return nothing.** That is the point of writing them down. An empty result is
this app's characteristic failure, and without a list saying which emptiness is correct, a deployment cannot
tell a rule that is working from a rule that is broken. The ratio has moved both ways, which is what makes it
worth stating: it improved as layers 3, 4 and 5 gained producers, and went the other way when the L2/L3 refresh
was found to have been empty all along under a note that credited it with 80 rows. Chain assembly is the one
worth following, because it has now been empty for three different reasons in turn: a missing `osi_layer`, then
lineage that never left one layer, and now a risk sum no pipeline event contributes to. Each fix made the next
blocker visible, which is what an expectation file is for.

## Where the sample events come from

[`make_sample_events.py`](./make_sample_events.py) runs the same `run_pipeline` the determinism tests call and
puts the output through the same `SiemWireStage` a deployment would put before its sink. Layers 1, 2 and 5 come
from the estate pipeline rather than the telemetry one, because a chain is decided by which classes were sealed
together: the same events rendered from the telemetry pipeline carry chains that stop at two layers. Layers 3
and 4 come from their own corpora and their own pipelines, and share an entity with neither the estate nor each
other -- an address is not a port, and a conversation is not an address. They are checked in so
that a change to what a SIEM would receive shows up in a pull request rather than only on a search head.

Regenerate after changing the corpus or a stage, and review the diff:

```bash
python examples/splunk_lineage_app/validate/make_sample_events.py
```

## What this does not establish

That a particular Splunk version parses the app the way `props.conf` says. This runs one version, in one
container, with everything on one instance. A real estate parses on indexers or heavy forwarders, and the
`KV_MODE` and acceleration choices in `props.conf` are estate-wide decisions this package does not make.
