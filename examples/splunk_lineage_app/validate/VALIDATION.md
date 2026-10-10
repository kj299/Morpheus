# Search-head validation

This package makes the search-head verdict cost minutes and produce a diff against a written expectation, rather
than a judgement call at the end of an afternoon of setup. It was first run on 2026-10-05, on Splunk 10.2.8: all
8,400 events indexed at their own times, every search ran, and forty-six of forty-eight returned what was written.
The two that did not were the wire's doing. Null fields were sent as `null`, Splunk did not read them as absent,
and `macs_per_port_step>0` let four null steps through R-B-L2-002 -- six ports where two had stepped. Binding
health had matched only because its expectation shared the defect: it counted four classes whose
`resolution_method` was null on every row as resolving bindings. The serializer now leaves null fields out, the
expectations say one row and two, and that run is kept in `search_head_runs/` as the evidence. The run over the
regenerated events, on 2026-10-09 on the same version, returned what was then written for all forty-eight, and is
kept beside it.

The third run, made the same day with the risk write path in place, used Splunk 10.2.8 installed from Splunk's own
tarball in the development container rather than the Docker image, and is kept in `search_head_runs/`. It is the
first in which the detections' rows outlive their jobs: every detection collects what it returns into
`behavior_risk`, and the run checks that the index holds exactly those rows -- 94 then -- before the searches that
read them run. It also found a search that could never have fired. R-P-L3-005 read the value two rows back as
`last(previous_destinations)` inside the `streamstats` creating `previous_destinations`, which Splunk evaluates as
null on every row; the second run's zero was that, and agreed with an expectation that blamed the summary instead.

The fourth, made that evening with this package exactly as shipped, on the Docker image and another machine,
returned what was then written for all forty-eight searches, held the same 94 risk records, and found the same
totals on the fifteen three-layer chains; it is kept in `search_head_runs/`, and `test_search_head_run.py` holds
the two installs to the same answer on every search and every check.

`search_head_results.json` holds the fifth, made late that night on the tarball install over the events
regenerated when each layer 5 principal gained a learned model of their own. It returned what is written for all
forty-eight searches: R-B-L5-001 and R-B-L5-002 return their first rows, 42 and 28, and the detections now write
166 risk records, so the run checks that the index holds exactly those rows -- 166 -- before the searches that
read them run. The fifteen three-layer chains carry the same risk as before, because every row the models add is
on a lineage of one layer. The four earlier runs differ from what is written now in exactly the four searches
the models changed, and the test names them.

## Running it

One command, on a machine with Docker and nothing else -- no Python, and no license file, because the image
starts under Splunk's built-in trial:

```bash
SPLUNK_PASSWORD='choose-one' examples/splunk_lineage_app/validate/run_search_head.sh
```

It starts the container, waits for it to report healthy, and runs [`run_search_head.py`](./run_search_head.py)
inside it under Splunk's own interpreter. That indexes every file in `sample_events/`, makes the two checks
below, dispatches all forty-eight searches in the order this document prescribes, and writes
`search_head_results.json` beside this file: the Splunk version, the date, what was indexed, a row count per
search, how many risk records the detections wrote and the index holds, and the risk on every chain that spans
three layers. Commit it; `tests/morpheus/determinism/test_search_head_run.py` compares it with
[`expected_results.json`](./expected_results.json). The container is left
running for rerunning a search by hand at <http://localhost:8000>; `docker compose down -v` here removes it.

Two adjustments make the run faithful, and both are the runner's rather than the reader's:

- **The events are dated 1970**, because the corpora count from the epoch, and no search head indexes that as
  itself: `MAX_DAYS_AGO` cannot exceed about thirty years, and an event outside it is silently given another time.
  The runner moves every timestamp forward by one whole number of weeks, so the newest event lands a few days
  before the run; hours of day, weekdays and every bin boundary stay where they were, and every search compares
  `_time` only with another `_time`. The events then span fifty-seven days, more than the thirty layer 1 allows,
  so the runner installs `validation_app/`, a separate app whose `local/props.conf` raises `MAX_DAYS_AGO` for
  this run and whose `local/indexes.conf` gives the indexes local paths. It is never deployed.
- **The searches look back from now**, over windows like `-2h@m` to `-5m@m`. Each is dispatched as
  `| savedsearch` with an explicit time range covering every event, which overrides the stanza's window, so each
  is asked its question over all the data at once -- the same simplification the Python recomputation behind
  `expected_results.json` makes.

The order is the one a schedule would settle into, made explicit: the binding refreshes, so the lookups exist; the
two predictive searches, so R-B-L7-002 reads a populated watchlist; every other detection, each collecting its rows
into `behavior_risk`; the behavior summary; R-P-L3-005, which reads the summary; Chain assembly, which sums the
risk; and the expiry jobs last, because the events are historical and expiry drops what they wrote. `collect`
hands its rows to the indexer and returns, so a search that reads an index another search wrote is dispatched only
once that index holds every row written to it. The runner also waits for an ordinary search, not only the
index-time count, to see every event before the first dispatch: on 2026-10-09 the two disagreed for long enough that
a summary dispatched in between returned 3,649 rows instead of 4,335. Then it rolls the hot buckets to warm, because
a count is still not enough: on one run every count reported layer 1's 305 events while R-D-L1-001 could not see the
one poll it needs, and returned that row a minute later. With the roll, three consecutive runs returned the same row
count for every search.

The indexing loop this section used to give by hand would have met both problems: it indexed the 1970 timestamps
as they were, so every event would have failed the first check below. Nothing in this repository records it
having been run against the files it named.

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
| R-D-L1-001, transceiver substitution | **1** | One port: `hq:sw1:Gi1/0/2`'s serial changed on a poll the flap count says the link never moved for. The other optic replaced this hour, on `Gi1/0/6`, is quiet because the device recorded the link dropping between the two polls, which is what a swap does. |
| R-P-L1-004, optical degradation forecast | **1** | One port, the failing optic on `Gi1/0/6`: the line through its readings gives it hours, and the search's one row per port carries the shortest time to the floor. The tap's step, the steady ports' jitter and the replacement optic project nothing. |
| R-D-L2-001, MAC count on an access port | **0** | Correct. Joins `port_designations`, which ships header-only; 11 candidate rows are waiting behind it. |
| R-B-L2-002, port-to-MAC binding novelty | **2** | The hub port, four above the one address it carried in every earlier snapshot, and the spoofed port, one above its own record: the two ports R-D-L2-001 would name, found without its designation list. Once each, because the next snapshot's baseline has absorbed the step. |
| R-D-L2-003, ARP anomaly | **1** | 24 contested observations aggregate to one notable on `10.0.0.1`. |
| R-D-L2-004, MAC in two places | **2** | A conflict at zero gap and a displacement at two seconds. The roaming device, displaced a full poll cadence later, is deliberately outside the threshold. |
| R-D-L2-005, authorization without authentication | **2** | One bypass on a quiet port, one that arrived while a legitimate exchange was open. |
| R-B-L3-001, fan-out expansion | **1** | One notable, for the scanner. The browser beside it reaches as many addresses and none of them inside the estate, which is the whole of the difference. |
| R-B-L3-002, beaconing | **6** | One notable from the layer 3 corpus, for the pair on a five-minute timer; the worker making plenty of flows to one file server at ragged intervals does not appear. Five more are R-C-002's hosts in the campaign corpus, each on a sixty-second timer -- the rule reports all five on their regularity alone. |
| R-D-L3-003, reserved-range egress | **1** | One flow, to `240.0.0.1`. What the number really asserts is that the other 410 flows are classified correctly. |
| R-B-L3-004, TTL fingerprint shift | **1** | One notable: twelve flows from one source, every one short by exactly the single hop an interposed device costs. The steady host beside it never moves. |
| R-P-L3-005, fan-out trajectory | **15** | Three for the planted scanner, whose fan-out climbs through the summary's five-minute bins, and two for each of six campaign sources whose fan-out rises three bins in a row. Empty until 2026-10-09, when a search head showed the search read a field its own `streamstats` was still creating; it now takes two passes. A watchlist rule, so fifteen is not fifteen pages. |
| R-D-L4-002, SYN without completion | **1** | One notable: 60 destination ports in one bin, none of them answering. The workstation's handshakes complete, and the sweep touches one port across thirty hosts rather than sixty on one. |
| R-D-L4-003, RST ratio | **2** | Two notables from one predicate. 50 flows sit at the same ratio; one server refusing 20 clients is an outage, 30 servers refusing one client is enumeration, and they need different responses. |
| R-B-L4-005, transfer envelope breach | **6** | One notable from the layer 4 corpus: 60000 bytes against the triple's own envelope of 1200. The busy triple beside it moves more in total and never breaches, which is what a global threshold could not express. Five more are the campaign's R-C-004 actors, each breaching its own sync envelope identically; which one was the exporting principal's, in session, is R-C-004's to say. |
| R-B-L6-001, new TLS client fingerprint | **5** | One notable, for the host that presented one stack across 34 handshakes and then a second. The host beside it is equally novel the first time its second stack appears, two handshakes in -- which is why the rule reads the settled-history floor and not novelty alone. Four more are R-C-002's settled hosts in the campaign corpus; its fifth, with five handshakes behind it, is held back by the same floor. |
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
| R-D-L5-007, off-hours authentication | **4** | The planted 03:00 sign-in by an office worker, a traveller's 08:00 sign-in before his flight, an hour his office week never used, and the first sign-ins at 15:00 and 17:00 on the afternoon another principal's account is used from Amsterdam. The service account signing in at 03:00 every night does not appear: the hour is ordinary in its own history. |
| R-D-L5-008, new authentication location or device | **2** | Two first sign-ins from New York by principals with mature histories: the impossible journey's far end and the traveller's arrival. The VPN user's first concentrator sign-in comes one sign-in into his history, before it is mature. It misses the account taken over from Amsterdam, whose first sign-in there failed and carried the first-seen flags the successes after it do not; the search reads successes only, and the model rules below find it. |
| R-D-L5-009, failed authentication run ending in success | **1** | Five refused attempts and then the approval, the same burst R-D-L5-004 reports from the factor's side. The fumbled password's two failures are below the threshold of three. |
| R-B-L5-001, composite authentication anomaly | **42** | Every row from a principal's own committed model, trained on the fortnight before the scored week. Twenty are the account taken over from Amsterdam: the twelve sign-ins of that afternoon, and eight more back in London, because the cumulative features stay raised until a model is trained on a window that includes it. The rest are the week's planted departures -- the fumbled password and the journey, the sign-in before a flight and the move to New York, the fatigue burst. The two principals whose week repeats their fortnight do not appear, and the joiner, scored by the population fallback, is kept out by the gate. The scores are uncalibrated: a feature the fortnight never varied scores any change in the thousands, so 6 and 2 read as "departed from the fortnight." |
| R-B-L5-002, location novelty anomaly | **28** | The three principals who reached somewhere new in the week, from the first row there on: twenty for the account taken over, six for the traveller, two for the impossible journey. The traveller's first is his sign-in from London before the flight, because the loss on `locincrement` is that feature's reconstruction error given all ten, not a flag for a new place. |
| R-B-L5-005, session duration anomaly | **1** | An office worker's eleven-hour day against four of eight. Her colleagues' eight-hour days measure at exactly their baseline. The service account's longest night would fire too, and is excluded because the identity context marks it a service account. |
| R-C-001, lateral movement chain | **1** | One chain, and the first chained rule here to return a row, because it reads scored events rather than notables: a fan-out rising to twenty-five new addresses -- half R-B-L3-001's threshold -- then a first login from that address to a server, then a process on the server no peer had run, inside thirty minutes. Six actors each miss one step and are quiet: the process before the login, the last step at thirty-five minutes, the principal's own server, a login from another address, a flat fan-out, and a server running only its routine. |
| R-C-004, staged exfiltration | **1** | One chain: a bulk export, twenty-five minutes later a fifty-times breach from the address the exporter's open session held, and twenty minutes after that a connection to a destination whose issuer nobody in the estate had seen. Four actors each do all three with one relation broken and are quiet: another principal's address, a breach after logging off, the corporate issuer, and the export last rather than first. |
| R-C-002, TLS before beaconing | **1** | One chain, and the first this rule has ever returned: it now reads the scored events rather than the two detections' notables. A settled host presents a new stack to a destination and fourteen minutes later its beacon there matures. Four hosts do both halves with one condition broken and are quiet: a beacon already running an hour before, a beacon to another address, a beacon maturing after sixty-seven minutes, and a host with five handshakes behind it. |
| R-C-005, credential replay across the stack | **1** | One principal at two switch ports 534 km apart twenty minutes apart: the leases name the workstations behind both sign-in addresses, and the MAC bindings closed from the two sites' switches put them at headquarters and in Edinburgh. Five others are one step short and quiet -- two ports at the same site, the same journey in three hours, an address no lease names, a lease that had ended an hour before, and a refused second attempt. Nothing here rests on geolocation: both ends of the journey are ports. |
| Behavior summary, per-layer scores | **4615** | One row per five-minute bin, layer, entity and lineage over the 8501 scored events. It returned nothing until `EnvelopeStampStage` put `osi_layer` and `entity_key` on every record, and 320 until the estate pipeline rendered these events with the desk authentications beside the ports; `peak_z` is still null outside layer 5's scored week, because only `TC5ScoreStage` produces `max_abs_z` and it does not score the fortnight the models were trained on. |
| Chain assembly, cross-layer risk | **0** | Correct, and now because the risk is written and is not enough. Every detection collects its rows into `behavior_risk` and this search sums them per lineage, once each. 15 of the 3591 chains span three layers -- a port's layer 1 samples, the layer 2 observations resolved onto it, and the authentications of the person sitting there -- and one detection accuses any of them: R-D-L2-003, at 55, under the 60 the search needs. The other 14 carry none; the run records both totals. `methods` reads every hop the resolver ladder took, so the 15 name `soft:directory`, `soft:dot1x` and `soft:mac_table` between them. |
| Binding lookup, L2/L3 refresh | **0** | Correct, and it always was. The search selects `binding_table=dhcp_lease`; this corpus has no DHCP source, and the 80 bucketed rows it used to be credited with are a MAC table under a different name. |
| Binding lookup, L1 refresh | **7** | Seven port intervals across five ports: three stable, two on each of the two ports whose optics are swapped. The lookup keys on port and switch with no bucket, so a swapped port's two collapse to one row and the later optic wins -- it answers what is in a port now, not what was in it then. |
| Binding lookup, L1 history refresh | **2** | Two rows, and two rows is the point. Only the ports whose optics were replaced have a superseded interval; the other three are described for all time by the current-state row and cost the history nothing. |
| Binding lookup, L1 history expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Principal watchlist, expiry | **2** | The runner dates the events so the newest lands three to ten days before the run. R-P-L7-006's two entries are stamped with their principals' latest events, which are the newest in the corpus, so they are still inside their thirty days; R-P-L5-006's six are stamped thirty-two days earlier and are dropped. It was written as 0 until the first search-head run showed that only the undated 1970 events expire everything. Run it last: run before R-B-L7-002, it empties the list that search reads. |
| Binding lookup, L2/L3 expiry | **0** | Correct. Nothing in a freshly loaded corpus is old enough to expire. |
| Binding health, unresolved rate | **1** | The ARP stream, the one class whose records carry a resolution outcome: 180 unresolved of 1,220 (0.148), under the 0.2 that marks a class degraded. It read five while the wire sent null fields: four classes carry `resolution_method` with no value, `resolution_method=*` matched the nulls, and the search reported them as resolving with nothing unresolved. An operational metric; the value matters, not whether it fired. |
| R-P-L5-006, drift trajectory | **6** | Two principals, neither of them behaviour, explained in `expected_results.json`: the two whose week repeats their fortnight climb on its last three days by hundredths a day, as the cadence features keep moving after their models were trained, and the rule measures that against a day-to-day spread that is nearly zero. The principals whose week does depart all cross the mean ceiling and leave. Watchlist, never a page: every firing is written to `principal_watchlist`. |

**Five of the forty-eight should return nothing.** That is the point of writing them down. An empty result is
this app's characteristic failure, and without a list saying which emptiness is correct, a deployment cannot
tell a rule that is working from a rule that is broken. The ratio has moved both ways, which is what makes it
worth stating: it improved as layers 3, 4, 5, 6 and 7 gained producers, and went the other way when the L2/L3 refresh
was found to have been empty all along under a note that credited it with 80 rows, and again when R-P-L3-005
turned out to have been empty because its own SPL could not fire. R-B-L5-001 and R-B-L5-002 left the list when
each principal gained a learned model of their own, which is what their gate had been waiting for. Chain assembly is the one worth following,
because it has now been empty for four different reasons in turn: a missing `osi_layer`, then lineage that never
left one layer, then a risk sum nothing wrote, and now a risk sum that is written and falls five points short on
the one three-layer chain a detection accuses. Each fix made the next blocker visible, which is what an
expectation file is for.

## The watchlist the predictive searches write

R-P-L5-006 and R-P-L7-006 write the `principal_watchlist` KV Store lookup as they run, and R-B-L7-002 reads
it. Run the two predictive searches first, then

```
| inputlookup principal_watchlist | stats count BY rule_id reason
```

should return two rows: `R-P-L5-006` with `drift-trajectory` and 6 entries, one per principal-day it fired
on, and `R-P-L7-006` with `access-breadth` and 2, one per principal. An empty lookup means the writers'
`outputlookup` was refused -- most often a collection missing from `collections.conf` on the search head -- and
R-B-L7-002 has been reading an empty list. None of the eight principals R-B-L7-002 reports is on the list, so
the severity of its notables does not move here; `test_saas_harness.py` asserts the weighting by putting a bulk exporter on it.
Run the expiry job last: it drops R-P-L5-006's entries, whose events are more than thirty days behind the run
once the runner has dated them, and keeps R-P-L7-006's two.

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
