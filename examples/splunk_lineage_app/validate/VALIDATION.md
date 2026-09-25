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
      context:*)                      idx=behavior_context ;;
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
| R-B-L4-005, transfer envelope breach | **6** | One notable from the layer 4 corpus: 60000 bytes against the triple's own envelope of 1200. The busy triple beside it moves more in total and never breaches, which is what a global threshold could not express. Five more are the campaign's R-C-004 actors, each breaching its own sync envelope identically; which one was the exporting principal's, in session, is R-C-004's to say. |
| R-B-L6-001, new TLS client fingerprint | **1** | One notable, for the host that presented one stack across 34 handshakes and then a second. The host beside it is equally novel the first time its second stack appears, two handshakes in -- which is why the rule reads the settled-history floor and not novelty alone. |
| R-D-L6-002, certificate issuer anomaly | **2** | Two notables: an interception, and one migration reported once. The delivery host rotating among four authorities differs from its own mode on 19 handshakes and the single-issuer gate removes all of them. |
| R-D-L6-003, self-signed to external destination | **1** | One notable over two connections, with a three-day certificate. Six of the corpus's eight self-signed handshakes are the internal appliance, which is ordinary and does not appear. |
| R-B-L6-004, cipher downgrade | **1** | One notable: a pair that negotiated modern suites ten times and then a broken one. The pair whose own suite varies routinely does not appear, and neither does the legacy appliance sitting at a low floor. |
| R-D-L6-005, content type mismatch | **1** | One notable: an archive behind a declared PNG. 50 handshakes reach the comparison and 49 are re-encodings, which is what comparing categories rather than types is for. |
| TLS table coverage, unrecognized values | **1** | An operational metric; the value matters, not whether it fired. Both counts are zero, which is what makes the five above mean what they claim. |
| R-B-L7-001, DNS tunneling | **1** | One notable: 130 random, long-labelled subdomains under one domain. The CDN fails only on label length, the tenant domain only on the count, and the SaaS provider on both per-query conditions -- each of the three conditions has a control it alone keeps quiet. |
| R-D-L7-005, enumeration | **2** | Two notables, one of which has found nothing and so has no ratio at all; it appears because the search tests 4xx > 0.7 * 2xx. The crawler fails only on the ratio and the broken client only on the path count. |
| R-B-L7-002, bulk data access | **8** | Eight notables at three severity levels, because the target's classification weights the notable rather than gating it: a restricted object at 75, a public one at 20, and one the inventory has never heard of at the default 40, flagged. The routine exporter, the one just under five times, the newcomer with no baseline and the denied export each fail exactly one condition. The other five are the campaign's R-C-004 actors, each exporting eighty times their baseline from an unclassified object, at 40. |
| R-P-L7-006, access breadth trajectory | **2** | Two principals watchlisted: one reaching a new object type every week with the same role throughout, and one whose role did change -- recorded four weeks late, so on every week of the rise the context said it had not. The principal with the same rise and a role change recorded on time is quiet. |
| R-B-L7-004, process ancestry novelty | **10** | Ten notables at three severity levels, because the integrity level weights the notable rather than gating it: Word starting PowerShell on a finance workstation at 55, although the build servers do it daily; a remote-access tool last run thirty-eight days earlier at 70; `mshta.exe` on a kiosk with no peer group at 40, flagged as judged on its own history; and `git` starting `curl` with no integrity level at the default 40. The compile a build server's peer runs daily, the editor under another user's profile, the kiosk's Notepad and the four-day-old lab machine each fail exactly one condition. The other six are the campaign corpus's servers, each starting a process at system integrity that no server in its group had run: R-B-L7-004 cannot tell the one that followed a fan-out and a first login from the five that did not, which is what R-C-001 is for. |
| R-D-L5-003, impossible travel | **2** | One principal in New York half an hour after her own London office login, and back in London ninety minutes later. Two rows is what one interloper produces. The eight-hour flight beside it, and the VPN user changing country twice a day, do not appear. |
| R-D-L5-004, multi-factor fatigue | **1** | Five denials in eight minutes and then an approval. The fumbled password beside it -- two failures and a success with the factor never challenged -- does not appear. |
| R-C-001, lateral movement chain | **1** | One chain, and the first chained rule here to return a row, because it reads scored events rather than notables: a fan-out rising to twenty-five new addresses -- half R-B-L3-001's threshold -- then a first login from that address to a server, then a process on the server no peer had run, inside thirty minutes. Six actors each miss one step and are quiet: the process before the login, the last step at thirty-five minutes, the principal's own server, a login from another address, a flat fan-out, and a server running only its routine. |
| R-C-004, staged exfiltration | **1** | One chain: a bulk export, twenty-five minutes later a fifty-times breach from the address the exporter's open session held, and twenty minutes after that a connection to a destination whose issuer nobody in the estate had seen. Four actors each do all three with one relation broken and are quiet: another principal's address, a breach after logging off, the corporate issuer, and the export last rather than first. |
| R-C-002, TLS before beaconing | **0** | Correct. Chains two detections' notables, and layer 6 has neither a producer nor a rule. It also reads `rule_id`, which detections write as they fire. |
| Behavior summary, per-layer scores | **4224** | One row per five-minute bin, layer, entity and lineage over the 7926 scored events. It returned nothing until `EnvelopeStampStage` put `osi_layer` and `entity_key` on every record, and 320 until the estate pipeline rendered these events with the desk authentications beside the ports; `peak_z` is still null outside layer 5, because only `TC5ScoreStage` produces `max_abs_z`. |
| Chain assembly, cross-layer risk | **0** | Correct, and for a new reason. The threshold the search was built around is met: 15 of the 3298 chains span three layers, holding a port's layer 1 samples, the layer 2 observations resolved onto it, and the authentications of the person sitting there. What stops it is the line after -- `total_risk >= 60 OR (layer_span >= 4 AND peak_z >= 4.0)` -- and `risk_score` is written into the index by the detection searches as they fire, not by a stage. This package indexes pipeline output alone, so every chain sums null. |
| Binding lookup, L2/L3 refresh | **0** | Correct, and it always was. The search selects `binding_table=dhcp_lease`; this corpus has no DHCP source, and the 80 bucketed rows it used to be credited with are a MAC table under a different name. |
| Binding lookup, L1 refresh | **5** | Five port intervals across four ports: three stable, two on the port whose optic is swapped. The lookup keys on port and switch with no bucket, so those two collapse to one row and the later optic wins -- it answers what is in a port now, not what was in it then. |
| Binding lookup, L1 history refresh | **1** | One row, and one row is the point. Only the port whose optic was replaced has a superseded interval; the other three are described for all time by the current-state row and cost the history nothing. |
| Binding lookup, L1 history expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Binding lookup, L2/L3 expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Binding health, unresolved rate | **1** | An operational metric; the value matters, not whether it fired. |
| R-P-L5-006, drift trajectory | **7** | Three principals, none of them behaviour, each explained in `expected_results.json`: two climb for six days because the reference scorer's baseline is frozen under cumulative features, one has a shallow run ended by the planted burst. Watchlist, never a page. |

**Seven of the thirty-seven should return nothing.** That is the point of writing them down. An empty result is
this app's characteristic failure, and without a list saying which emptiness is correct, a deployment cannot
tell a rule that is working from a rule that is broken. The ratio has moved both ways, which is what makes it
worth stating: it improved as layers 3, 4, 5, 6 and 7 gained producers, and went the other way when the L2/L3 refresh
was found to have been empty all along under a note that credited it with 80 rows. Chain assembly is the one
worth following, because it has now been empty for three different reasons in turn: a missing `osi_layer`, then
lineage that never left one layer, and now a risk sum no pipeline event contributes to. Each fix made the next
blocker visible, which is what an expectation file is for.

## Where the sample events come from

[`make_sample_events.py`](./make_sample_events.py) runs the same `run_pipeline` the determinism tests call and
puts the output through the same `SiemWireStage` a deployment would put before its sink. Layers 1, 2 and 5 come
from the estate pipeline rather than the telemetry one, because a chain is decided by which classes were sealed
together: the same events rendered from the telemetry pipeline carry chains that stop at two layers. Layers 3
4, 6 and 7 come from their own corpora and their own pipelines, and share an entity with neither the estate
nor each other -- an address is not a port, a conversation is not an address, and the host a handshake or a query
was made from is not any of them. They are checked in so
that a change to what a SIEM would receive shows up in a pull request rather than only on a search head.

Regenerate after changing the corpus or a stage, and review the diff:

```bash
python examples/splunk_lineage_app/validate/make_sample_events.py
```

## What this does not establish

That a particular Splunk version parses the app the way `props.conf` says. This runs one version, in one
container, with everything on one instance. A real estate parses on indexers or heavy forwarders, and the
`KV_MODE` and acceleration choices in `props.conf` are estate-wide decisions this package does not make.
