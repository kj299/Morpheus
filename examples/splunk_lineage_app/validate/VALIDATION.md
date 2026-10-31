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

The fifth, made late that night on the tarball install over the events regenerated when each layer 5 principal
gained a learned model of their own, returned what was then written for all forty-eight searches: R-B-L5-001 and
R-B-L5-002 return their first rows, 42 and 28, and the detections wrote 166 risk records. It is kept in
`search_head_runs/`.

The sixth, made on 2026-10-10 on the same install over the events step 9 regenerated, when layer 3 gained a
fortnight of ordinary history and each host a baseline of its own, and the app four more searches, returned what
was then written for all fifty-two, with 198 risk records. A run an hour before it, over the first draft of step
9's events, returned eight beaconing pairs where six were written: the new corpus's port sweep and a conferencing
client's media streams had been planted on a metronome, so R-B-L3-002 read both as beacons, and the package's own
derivation found the same two. They now keep the uneven pace real ones do, and that run is not kept. The sixth is
kept in `search_head_runs/`.

The seventh, made later on 2026-10-10 on the same install over the events step 10 regenerated, when the estate gained the network objects' plants -- a tap, a flapping link, a restarted switch, a
climbing error count, a re-patch and an inserted device, a surge and a silence, a camera VLAN, two lab benches
timing their 802.1X exchanges -- and the app ten more searches. It returned what is written for all sixty-two,
over 10,285 indexed events: each of the ten returns what its plant leaves for it, and the detections now write 209
risk records, so the run checks that the index holds exactly those rows -- 209 -- before the searches that read
them run. One of the fifteen
three-layer chains now carries 50, from the tap on its port, beside the one at 55; neither reaches 60. The six
earlier runs differ from what is written now in exactly the searches the models, the host baselines and step 10's
corpus changed, and the test names them. It is kept in `search_head_runs/`.

The eighth, made that night on the same install over the events step 11 regenerated, when the pipeline began
reporting on itself: three more sourcetypes in three more indexes -- 851 health records, the DMZ poller's three
quarantined polls and the layer 3 corpus's one late flow -- and a second SaaS tenant whose sync doubles. It returned
what was then written for all sixty-six over 11,352 indexed events, and the detections wrote the same 209 risk
records as the seventh: every detection now reads `window_complete=true`, and no scored event is incomplete, so the
filter changed no answer. The four new searches return one row each -- the DMZ poller's refusal rate, the ARP
sensor's three missing sequence numbers, the late flow, and the labs tenant's step -- and the seventh run differs
from what is written only in the summary, which groups the second tenant's exports.

The ninth, made on 2026-10-15 on the same install over the events step 12
regenerated, when hosts gained an identity across layers: 1505 lease buckets, 803 more MAC buckets and 59 supplicant
buckets on `binding:bucketed`, and 2751 edges on `morpheus:edge` in place of 47 flow records. It returned what was
then written for all seventy-two over 16,423 indexed events, with the same 209 risk records: the L2/L3 refresh its first
rows, 1505; the host evidence report seventeen machines; and the identity walk all twenty-five sign-ins, each to the
port the pipeline resolved. Its first attempt found the walk empty: the refreshes keyed each row on the bucket
ordinal the producer wrote, counted from 1970, while the walk computed its bucket from the event's `_time`, which
the run moves by whole weeks. Every refresh now keys its rows on the bucket its own time falls in, which is the same
number wherever the time is the one the producer wrote, and the expiry jobs, which had kept nothing only because
those ordinals were half a century old, keep every row. The eighth run differs from what is written exactly there.
It is kept in `search_head_runs/`.

The tenth was made on 2026-10-16 on the same install over the events step 13
regenerated, when a service and an application became entities: a week of DNS and HTTP history per client, 303
service connections, the hosts whose stacks R-B-L4-004 compares with their browsers, a shared address fronting two
names, and the three records of `context:application`. It returned what was then written for all seventy-five over
20,127 indexed events, with 213 risk records. The three new detections return one row each; R-D-L6-002
returns three, the shared address's interception among them; and R-B-L7-001 and R-D-L7-005 return what they did,
the reputation agent and the scanner each kept out by its own busiest day. The host evidence report counts
nineteen machines where seventeen were written: the expectation had not been derived from the events, and two of
R-B-L4-004's hosts are seen at layers 4 and 7. The package derives it now. The ninth run differs from what is written
in those two searches and the summary. The tenth is kept in `search_head_runs/` too.

The eleventh, made on 2026-10-17 on the same install over the events step 15 regenerated, when the switch and the
router became entities and R-C-003 got a chain to find: 305 device polls beside the estate's port polls, and the
campaign's headquarters desk ports -- 292 port polls and 1125 MAC table walks on layers 1 and 2, which the campaign
had not sent before -- with four intruders' leases and flows. It returned what was then written for all seventy-eight
over 22,410 indexed events, and the detections wrote 221 risk records. The three new detections returned one row
each. R-B-L2-002 and R-B-L2-006 each gained the campaign's desk ports: the rogue host is a second address on its port
and a vendor new to its VLAN, and so are its controls, which is why the chain exists. The restart report reads the
devices' own polls, where the storm switch restarts in three periods. The tenth run differs from what is written in
those three searches, the two lower binding rungs and the jobs that expire them, and the summary. The eleventh is
kept in `search_head_runs/`.

The twelfth, made on 2026-10-22 on the same install over the events regenerated when the host segment gained a
sourcetype of its own: 1568 host frames, one per workstation per hour over three weeks, scored against the committed
peer-group models, on `morpheus:score:host`. It returned what was then written for all eighty over 24,146 indexed
events, and the detections wrote 224 risk records. R-P-L3-009 returned its three host-days and wrote them to
`host_watchlist`, whose expiry job kept all three; the summary grew by the 1568 frames, each its own group, and
`peak_z` is a number on the 536 scored ones; and no other search moved, because the host corpus's machines share
nothing with any other corpus's. The eleventh run differs from what is written in the summary and R-B-L6-001, and
has none of the four searches the host half added. The twelfth is kept in `search_head_runs/` too.

The thirteenth, kept in `search_head_runs/` with the twelve before it, was made on 2026-10-23 on the same install over
the events regenerated when the daily drift reached the TLS and DNS columns: a week of handshakes from a host that acquires a stack a day and
one that acquires two and stops, and eight days of a client tunnelling slowly under R-B-L7-001's floor. It returned
what is written for all eighty-two over 24,736 indexed events, and the detections wrote 233 risk records, so the run
checks that the index holds exactly those rows -- 233 -- before the searches that read them run. R-P-L6-006 returns
the creeping host's last three days and R-P-L7-009 the slow tunnel's last four, and both write `host_watchlist`: the
run reads the list after the predictive searches and finds three rows, R-P-L3-009's three entries, R-P-L6-006's three
and R-P-L7-009's four, and the expiry job keeps R-P-L3-009's and drops the seven whose days sit in the first week of
1970's timeline. R-B-L6-001 grows from five rows to seven, the creeping host and its control, and the summary by the
584 scored events; R-B-L7-001 returns its one tunnel as before, the slow one never reaching its count. The twelfth
run differs from what is written in the summary and R-B-L6-001, and has neither of the two searches this part added.
The thirteenth differs from what is written in Chain assembly alone: the desk spoof planted on 2026-10-24 put the
ARP anomaly's 55 on the tapped port's chain beside the tap's 50, so the search now expects the one row the run before
the plant could not have returned; the anomaly's own notable count is unchanged, since the search groups by address.

The fourteenth, kept in `search_head_runs/` with the thirteen before it, was made on 2026-10-26 on the same install over the
events regenerated for the desk spoof and for plan section 3.2, which added the network inventory on
`context:network` and R-D-L1-010. It returned what was then written for all eighty-three over 24,766 indexed
events, and the detections wrote 235 risk records, so the run checks that the index holds exactly those rows --
235 -- before the searches that read them run. R-D-L1-010 returns its two ports, the tapped desk and the failing optic R-D-L1-002 cannot
see; Chain assembly returns the one chain, at 135, with R-D-L1-002's 50, R-D-L1-010's 30 and R-D-L2-003's 55 on it,
beside the flooding host's chain at 55; and the twenty inventory records index as `context:network` and are read by
no search, because the scores carry what they say. It differs from what is now written in the summary alone, and has
neither of the two searches plan section 3.3 added.

The fifteenth, kept as `search_head_runs/2026-10-28T1645Z.json`, was made on 2026-10-28 on the same install over the
events regenerated for plan section 3.3, which gave the HTTP and endpoint classes a daily trajectory each, the slow
enumerator and the creeping build server to read, and R-P-L7-010 and R-P-L7-011. It returned what was then written
for all eighty-five, and the detections wrote 242 risk records. R-P-L7-010 returns the slow enumerator's
last four days and R-P-L7-011 the build server's three, and both write `host_watchlist`: the run reads the list after
the predictive searches and finds five rows, and R-B-L7-004 raises the build server's `curl` from 60 to 75 for the
entry, which no row count shows and the risk index does. The summary grew by the 865 new requests and process starts,
to 9032 rows. It differs from what is now written only by lacking the search plan section 3.7 added.

The sixteenth, `search_head_results.json` beside this file, was made on 2026-10-31 on Splunk 10.2.8 unpacked from the
official tarball and run as its own user, not in the container, over the events regenerated for plan section 3.7,
whose health records answer the fallback count from each window's own rows. It returned what is written
for all eighty-six over 25,632 indexed events, and the detections wrote 242 risk records, so the run checks that
the index holds exactly those rows -- 242 -- before the searches that read them run. `Pipeline health - stage
counts` returns its five classes. Every risk record carries the `thresholds_hash` plan section 3.6 added, and a query
over `behavior_risk` after the run found all 242 equal to their rule's entry in `rule_metadata`, over the sixty rules
that fired.

## Running it

One command, on a machine with Docker and nothing else -- no Python, and no license file, because the image
starts under Splunk's built-in trial:

```bash
SPLUNK_PASSWORD='choose-one' examples/splunk_lineage_app/validate/run_search_head.sh
```

It starts the container, waits for it to report healthy, and runs [`run_search_head.py`](./run_search_head.py)
inside it under Splunk's own interpreter. That indexes every file in `sample_events/`, makes the two checks
below, dispatches all eighty-six searches in the order this document prescribes, and writes
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
  `_time` only with another `_time`. The events then span sixty days, more than the thirty the app allows edges,
  quarantined records and health records, so the runner installs `validation_app/`, a separate app whose `local/props.conf` raises `MAX_DAYS_AGO` for
  this run and whose `local/indexes.conf` gives the indexes local paths. It is never deployed.
- **The searches look back from now**, over windows like `-2h@m` to `-5m@m`. Each is dispatched as
  `| savedsearch` with an explicit time range covering every event, which overrides the stanza's window, so each
  is asked its question over all the data at once -- the same simplification the Python recomputation behind
  `expected_results.json` makes.

The order is the one a schedule would settle into, made explicit: the binding refreshes, so the lookups exist; the
five predictive searches, so R-B-L7-002 and R-B-L7-004 read populated watchlists; every other detection, each collecting its rows
into `behavior_risk`; the behavior summary; R-P-L3-005, which reads the summary; Chain assembly, which sums the
risk; and the expiry jobs last, because each rewrites the lookup it trims and one run earlier could take away what a
later search reads. `collect` hands its rows to the indexer and returns, so a search that reads an index another
search wrote is dispatched only once that index holds every row written to it. The runner also waits for an ordinary
search, not only the index-time count, to see every event before the first dispatch: on 2026-10-09 the two disagreed
for long enough that a summary dispatched in between returned 3,649 rows instead of 4,335. Then it rolls the hot
buckets to warm, because a count is still not enough: on one run every count reported layer 1's 305 events while
R-D-L1-001 could not see the one poll it needs, and returned that row a minute later. With the roll, three
consecutive runs returned the same row count for every search.

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
   `index=behavior_bindings sourcetype=binding:bucketed | stats count BY binding_table`. Four rows here:
   `dhcp_lease` with 1685 events, `dot1x_supplicant` with 59, `mac_table` with 1249 and `port_inventory` with 7.
   The 2026-10-23 run records the same four. An empty result means the refresh search selects on a field the producer
   stopped writing, and the lookup silently stops being refreshed.

## What each search should return

Full detail with reasons in [`expected_results.json`](./expected_results.json). Summary, including the ones that
should return nothing:

| Search | Rows | Note |
|---|---|---|
| R-D-L1-001, transceiver substitution | **1** | One port: `hq:sw1:Gi1/0/2`'s serial changed on a poll the flap count says the link never moved for. The other optic replaced this hour, on `Gi1/0/6`, is quiet because the device recorded the link dropping between the two polls, which is what a swap does. |
| R-P-L1-004, optical degradation forecast | **1** | One port, the failing optic on `Gi1/0/6`: the line through its readings gives it hours, and the search's one row per port carries the shortest time to the floor. The tap's step, the steady ports' jitter and the replacement optic project nothing. |
| R-D-L1-002, optical tap step | **1** | One port, `Gi1/0/3`, which loses three decibels of receive light at minute forty while its transmit level holds: twenty-one polls a decibel or more under the port's own median, one notable. The failing optic slides with its own median and never reads more than 0.867 dB under it, which makes it R-P-L1-004's; the two replaced optics change serial and are excluded on that poll. |
| R-D-L1-003, link flap instability | **1** | One port, `Gi1/0/7`, down and back in every polling gap from the half hour: ten transitions in its hour on the fifth flapping poll. The single flap, the swap, the re-patch and the insertion are two transitions each, and the restarted switch's are labelled a device reset and excluded. |
| Device restart | **4** | One row for `sw2`, which restarts at minute thirty: its own poll reads the uptime gone backwards, and its first poll back reports a counter reset and a device reset on each of its three ports -- three flags and the device's own restart, one device. Three more for `sw5`, polled only as a device, one for each five-minute period of its reboot storm. An operational report, so it writes no risk. |
| R-B-L1-005, error rate step | **1** | One port, `Gi1/0/8`, whose CRC errors climb by thirty more each minute from minute forty-five: sixteen polls more than 0.1 errors a second above the highest five-minute peak of its own history. Every other port stays within 0.0167 of its record. |
| R-D-L1-006, LLDP neighbor change | **2** | The re-patched port, `Gi1/0/11`, from `dist-sw3` to `dist-sw4`, and `Gi1/0/12`, which reports a neighbour the estate has never seen: a device inserted in the path. The restarted switch's ports report no neighbour for one poll and then their old one, which is no neighbour they have not had. |
| R-B-L1-007, interface volume departure | **2** | `Gi1/0/9` sends nine hundred megabits a second where it never carried more than about sixty, and `Gi1/0/10` carries nothing with its link up, each for the five polls until the next period joins the history. Every other port carries what it always has. |
| R-D-L1-008, configuration change outside the change window | **1** | `sw1`'s configuration changed at minute forty, outside the window of the hour's first ten minutes, to one it had never run. The push at minute two is inside the window and does not fire. |
| R-P-L1-009, reboot-rate trajectory | **1** | `sw5`'s third restart in its third five-minute period, each period holding more restarts than the one before. `sw2`'s single restart is a run of one. |
| R-D-L1-010, optical level below commissioning | **2** | Two ports read against the `-7 dBm` the network inventory records their links commissioned at: the tap on `Gi1/0/3`, twenty-one polls from minute forty, the deepest 3.048 dB under; and the failing optic on `Gi1/0/6`, six polls from minute thirty-nine until it is replaced, the deepest 1.756. The second is the case R-D-L1-002 cannot see, because its median slides down with the optic. The new optic's poll is excluded and the port is recommissioned from the swap; `Gi1/0/10`, which the inventory never recorded, has no level and is not read. |
| Topology, LLDP adjacency | **15** | One link per polled port, thirteen, and one more on each of the two ports whose neighbour changed. A port with no neighbour on a poll writes no link rather than a link to nobody, and a second run writes the same fifteen. A lookup refresh, not a detection. |
| R-D-L2-001, MAC count on an access port | **0** | Correct. Joins `port_designations`, which ships header-only; 17 candidate rows are waiting behind it. |
| R-B-L2-002, port-to-MAC binding novelty | **6** | The hub port, four above the one address it carried in every earlier snapshot, and the spoofed port, one above its own record: the two ports R-D-L2-001 would name, found without its designation list, which this rule reads only to leave trunks and LAG members out. Once each, because the next snapshot's baseline has absorbed the step. The other four are the campaign's headquarters desk ports, each one above its record when an intruder plugged in: R-C-003's attacker and its three controls, which this rule cannot tell apart. |
| R-D-L2-003, ARP anomaly | **1** | 24 contested observations aggregate to one notable on `10.0.0.1`. |
| R-D-L2-004, MAC in two places | **2** | A conflict at zero gap and a displacement at two seconds. The roaming device, displaced a full poll cadence later, is deliberately outside the threshold. |
| R-D-L2-005, authorization without authentication | **2** | One bypass on a quiet port, one that arrived while a legitimate exchange was open. |
| R-B-L2-006, new vendor on a VLAN | **3** | The camera VLAN, `hq:20`, when a single-board computer appears on it, `hq:10`, when the hub's MACs arrive, and the campaign's desk VLAN, `hq:30`, when R-C-003's single-board computer and its printer arrive, two new vendors in one notable. Keyed on site and VLAN, so the same VLAN number at another site is another segment. The third camera, of the vendor the camera VLAN already carries, does not step. |
| R-B-L2-007, slow 802.1X authorization | **1** | The first lab bench's exchange at minute forty-five: three attempts, and nine seconds after the last, three times the slowest of its port's prior exchanges. The desk ports reauthenticate every fifteen minutes and never have the twenty prior exchanges a distribution needs inside the hour. |
| R-B-L2-008, instant 802.1X authorization | **1** | The second lab bench's exchange accepted in the second it was requested, where each before it took four. The two bypasses carry no elapsed time at all, which is R-D-L2-005's signal. |
| R-B-L3-001, fan-out expansion | **1** | One notable, for the scanner, measured against its own fortnight: it reached three servers an hour for fourteen days and eighty addresses in its last scored hour. The DHCP server's Monday checks reach sixty, which the literal this rule shipped with fired on, and its own Mondays reach sixty-four, so it is no step; the browser reaches as many addresses and none of them inside the estate. |
| R-B-L3-002, beaconing | **6** | One notable from the layer 3 corpus, for the pair on a five-minute timer; the worker making plenty of flows to one file server at ragged intervals does not appear. Five more are R-C-002's hosts in the campaign corpus, each on a sixty-second timer -- the rule reports all five on their regularity alone. |
| R-D-L3-003, reserved-range egress | **1** | One flow, to `240.0.0.1`. What the number really asserts is that the other 1,183 flows are classified correctly. |
| R-B-L3-004, TTL fingerprint shift | **1** | One notable: twelve flows from one source, every one short by exactly the single hop an interposed device costs. The steady host beside it never moves. |
| R-P-L3-005, fan-out trajectory | **43** | Four for the planted scan, fourteen for the same workstation's ordinary mornings, eleven for the DHCP server's Monday checks, and two for each of seven campaign sources whose fan-out rises three bins in a row: the rule reads shape and not history, so a host's first minutes of the day read as an expansion. Empty until 2026-10-09, when a search head showed the search read a field its own `streamstats` was still creating; it now takes two passes. A watchlist rule, so forty-three is not forty-three pages. |
| R-B-L3-006, fan-in onto a workstation | **1** | One notable: a finance workstation reached by fourteen internal sources where its fortnight held one. The DNS server's ordinary Mondays are busier than this one, and the application server steps when a new service is pointed at it and is a server. |
| R-B-L3-007, first-contact upload | **1** | One notable: nine megabytes to a network the host had never reached, against a fortnight whose most lopsided hour was its nightly backup. The first download from another new network, and the backup provider sent more than ever, are each one half and quiet. |
| R-D-L3-008, port sweep | **1** | One notable: forty ports on one internal server. A conferencing client spreads across thirty ports on an external relay, and the horizontal scan reaches eighty addresses on one port. |
| R-P-L3-009, host drift trajectory | **3** | One host, the engineering workstation planted to drift, on the last three of its seven scored days: its daily mean against its peer group's model climbs every scored day, from 0.7 to 379, and the rule reports it from the first day the drift stage calls the trajectory mature, the fifth rising day, not the fourth that first meets the conditions. No ceiling: the host scores run into the hundreds and the acceleration into the tens, which R-P-L5-006's ceilings would read as not a trajectory. Watchlist, never a page: every firing is written to `host_watchlist`, which R-B-L7-004 reads. |
| Connection evidence, layers 3, 4 and 6 | **1** | One connection: the campaign attacker's breach, which the flow exporter, the packet sensor and the inspection point each stamped with one Community ID. The four other breaches were seen at two layers, as was the connection R-C-002's attacker presented its new fingerprint on. A report, not a detection. |
| Host evidence, one machine across layers | **19** | Nineteen machines under one `host_key` at two layers or more: the campaign's ten workstations, resolved from their addresses through the DHCP server's leases and the EDR inventory at layers 3, 4 and 6; seven clients the network and transport corpora both carry, which resolve the same way and so are the same workstations; and two of R-B-L4-004's hosts, whose packets and requests both resolve to them, at layers 4 and 7. The campaign's servers have static addresses no lease names, and their processes are seen at layer 7 alone. A report, not a detection. |
| Identity walk, sign-in to switch port | **25** | Twenty-five sign-ins walked to a port through the lookups, each to the port the pipeline resolved: the estate's fifteen desk sign-ins by their 802.1X identity, and ten campaign host logins by address to a MAC and the MAC to a port. A sign-in from an address no lease named at the time is walked to nothing, as the pipeline resolved it to nothing. Run after the three refreshes. A report, not a detection. |
| R-D-L4-002, SYN without completion | **1** | One notable: 60 destination ports in one bin, none of them answering. The workstation's handshakes complete, and the sweep touches one port across thirty hosts rather than sixty on one. |
| R-D-L4-003, RST ratio | **2** | Two notables from one predicate. 50 flows sit at the same ratio; one server refusing 20 clients is an outage, 30 servers refusing one client is enumeration, and they need different responses. |
| R-B-L4-004, stack fingerprint mismatch | **1** | One host, `ws-0-61`: its requests claim Chrome on Windows and in its second hour its SYN packets carry a Linux stack's option order, new to it. The join is on `host_key`, the one entity layers 4 and 7 share. Its neighbour runs the same browser over a Windows stack throughout, and the host running curl claims no operating system for its stack to contradict. |
| R-B-L4-005, transfer envelope breach | **6** | One notable from the layer 4 corpus: 60000 bytes against the triple's own envelope of 1200. The busy triple beside it moves more in total and never breaches, which is what a global threshold could not express. Five more are the campaign's R-C-004 actors, each breaching its own sync envelope identically; which one was the exporting principal's, in session, is R-C-004's to say. |
| R-B-L6-001, new TLS client fingerprint | **7** | One notable, for the managed host that presented one stack across 34 handshakes and then a second. The host beside it is equally novel the first time its second stack appears, two handshakes in -- which is why the rule reads the settled-history floor and not novelty alone -- and a personal device does exactly what the managed host does and is in no inventory, so the managed-endpoint gate holds it back. Two more are the week R-P-L6-006 reads: the host acquiring a stack a day fires on each of its six acquisitions, and the one that acquires two and stops on both; this rule says each stack is new, and nothing about the pace. Four more are R-C-002's settled hosts in the campaign corpus; its fifth, with five handshakes behind it, is held back by the floor. |
| R-D-L6-002, certificate issuer anomaly | **3** | Three notables, one per TLS destination -- the address and the server name together: an interception, one migration reported once, and an interception of one of two names a shared address fronts. Keyed on the address alone, that host has presented two authorities from the start and the single-issuer gate never opens on it. The delivery host rotating among four authorities differs from its own mode on 19 handshakes and the gate removes all of them. |
| R-D-L6-003, self-signed to external destination | **1** | One notable over two connections, with a three-day certificate. Six of the corpus's eight self-signed handshakes are the internal appliance, which is ordinary and does not appear. |
| R-B-L6-004, cipher downgrade | **1** | One notable: a pair that negotiated modern suites ten times and then a broken one. The pair whose own suite varies routinely does not appear, and neither does the legacy appliance sitting at a low floor. |
| R-D-L6-005, content type mismatch | **1** | One notable: an archive behind a declared PNG. 50 handshakes reach the comparison and 49 are re-encodings, which is what comparing categories rather than types is for. |
| R-P-L6-006, client stack trajectory | **3** | One host, the workstation planted to acquire a TLS stack a day, on the last three of its seven days: its distinct-stack count climbs one a day under the daily seal, and the rule reports it from the fifth day, the first the drift stage calls mature, as R-P-L3-009 does. The host that acquires two stacks and stops rises for three days, one short, and is quiet; every other host presents the same stacks day after day. Watchlist, never a page. |
| TLS table coverage, unrecognized values | **1** | An operational metric; the value matters, not whether it fired. Both counts are zero, which is what makes the five above mean what they claim. |
| R-B-L7-001, DNS tunneling | **1** | One notable: 129 random, long-labelled subdomains under one domain in one hour, 130 times the client's busiest day. The CDN fails only on label length, the tenant domain only on the count, the SaaS provider on both per-query conditions, and a reputation agent looking up file hashes as subdomains only on its baseline: 120 on the event day against 100 every day before it -- each of the four conditions has a control it alone keeps quiet. The slow tunnel R-P-L7-009 watchlists never reaches the count: 52 subdomains on its busiest day, half the floor. |
| R-D-L7-005, enumeration | **2** | Two notables, one of which has found nothing and so has no ratio at all; it appears because the search tests 4xx > 0.7 * 2xx. The crawler fails only on the ratio, the broken client only on the path count, and an authorized scanner requesting 220 paths against 200 a day only on its baseline. |
| R-B-L7-007, service reached by new clients | **1** | One service, `payroll-api.corp.example:443`, behind the reverse proxy: fifteen clients in an hour where three a day was its busiest, one on client software it had never seen. The wiki on the same address and port grows as much on familiar browsers, the unnamed metrics port gains a fingerprint without gaining clients, the CI server gains a tool and too few clients, and a service on its first day has no baseline. |
| R-B-L7-008, domain generation | **1** | One client: 80 registered domains in an hour, nine in ten NXDOMAIN, after a week of six a day. A busy browser reaches 70 domains that all resolve, and a link checker finding most of 60 dead is 1.2 times its own busiest day. |
| R-B-L7-002, bulk data access | **8** | Eight notables at three severity levels, because the target's classification weights the notable rather than gating it: a restricted object at 75, a public one at 20, and one the inventory has never heard of at the default 40, flagged. The routine exporter, the one just under five times, the newcomer with no baseline and the denied export each fail exactly one condition. The other five are the campaign's R-C-004 actors, each exporting eighty times their baseline from an unclassified object, at 40. |
| R-P-L7-006, access breadth trajectory | **2** | Two principals watchlisted: one reaching a new object type every week with the same role throughout, and one whose role did change -- recorded four weeks late, so on every week of the rise the context said it had not. The principal with the same rise and a role change recorded on time is quiet. |
| R-P-L7-009, subdomain count trajectory | **4** | One client, the one planted to tunnel slowly, on the last four of its eight days: ten random subdomains of one domain on its first day and six more each day after, 52 on the last, which R-B-L7-001 never sees because 52 is half its floor and 1.13 times the day before. The daily seal carries each day's largest count, the rule reads it from the fifth day, and the two rules read the same column. The loud tunnel's client rises for two days and the reputation agent's for two; neither watchlists. |
| R-P-L7-010, path count trajectory | **4** | One client, the one planted to probe slowly, on the last four of its eight days: forty paths on its first morning and ten more each morning after, 110 on the last, which R-D-L7-005 never sees because 110 is ninety short of its floor and 1.1 times its busiest day. The rule reads it from the fifth day, as R-P-L7-009 reads the slow tunnel, and each row carries the forecast: the days its line takes to reach R-D-L7-005's two hundred, twelve on the fifth day and nine on the last. The scanner's mornings are as busy as each other and never rise. |
| R-P-L7-011, process breadth trajectory | **3** | One host, the second build server, on the last three days of its creep: from day 27 it takes up one more each day of the five tool pairs its peer runs every morning, four distinct pairs a day becoming nine. R-B-L7-004 is quiet on every one of them, because its peer group runs them all; the breadth is what moves. The rule reads it from the creep's third day, the first with four rising days. No other host widens. |
| R-B-L7-004, process ancestry novelty | **11** | Eleven notables at five severity levels, because the integrity level and the host's criticality weight the notable rather than gating it: Word starting PowerShell on a medium finance workstation at 55, although the build servers do it daily, on a workstation whose EDR has spelled it `FIN-01` since day 20 and is one host throughout; a remote-access tool last run thirty-eight days earlier at 70; `mshta.exe` on a low-criticality kiosk with no peer group at 20, flagged as judged on its own history; and `git` starting `curl` with no integrity level on a high-criticality build server at 60, raised to 75 because R-P-L7-011 has put that build server on the hosts' list for widening. The compile a build server's peer runs daily, the editor under another user's profile, the kiosk's Notepad and the four-day-old lab machine each fail exactly one condition. The other seven are the campaign corpus's: six high-criticality servers each starting a process at system integrity that no server in its group had run, 105 apiece, and the finance workstation one actor reached, 70. R-B-L7-004 cannot tell the one that followed a fan-out and a first login to a server from the others, which is what R-C-001 is for. |
| R-D-L5-003, impossible travel | **2** | One principal in New York half an hour after her own London office login, and back in London ninety minutes later. Two rows is what one interloper produces. The eight-hour flight beside it, and the VPN user changing country twice a day, do not appear. |
| R-D-L5-004, multi-factor fatigue | **1** | Five denials in eight minutes and then an approval. The fumbled password beside it -- two failures and a success with the factor never challenged -- does not appear. |
| R-D-L5-007, off-hours authentication | **4** | The planted 03:00 sign-in by an office worker, a traveller's 08:00 sign-in before his flight, an hour his office week never used, and the first sign-ins at 15:00 and 17:00 on the afternoon another principal's account is used from Amsterdam. The service account signing in at 03:00 every night does not appear: the hour is ordinary in its own history. |
| R-D-L5-008, new authentication location or device | **2** | Two first sign-ins from New York by principals with mature histories: the impossible journey's far end and the traveller's arrival. The VPN user's first concentrator sign-in comes one sign-in into his history, before it is mature. It misses the account taken over from Amsterdam, whose first sign-in there failed and carried the first-seen flags the successes after it do not; the search reads successes only, and the model rules below find it. |
| R-D-L5-009, failed authentication run ending in success | **1** | Five refused attempts and then the approval, the same burst R-D-L5-004 reports from the factor's side. The fumbled password's two failures are below the threshold of three. |
| R-B-L5-001, composite authentication anomaly | **42** | Every row from a principal's own committed model, trained on the fortnight before the scored week. Twenty are the account taken over from Amsterdam: the twelve sign-ins of that afternoon, and eight more back in London, because the cumulative features stay raised until a model is trained on a window that includes it. The rest are the week's planted departures -- the fumbled password and the journey, the sign-in before a flight and the move to New York, the fatigue burst. The two principals whose week repeats their fortnight do not appear, and the joiner, scored by the population fallback, is kept out by the gate. The scores are uncalibrated: a feature the fortnight never varied scores any change in the thousands, so 6 and 2 read as "departed from the fortnight." |
| R-B-L5-002, location novelty anomaly | **28** | The three principals who reached somewhere new in the week, from the first row there on: twenty for the account taken over, six for the traveller, two for the impossible journey. The traveller's first is his sign-in from London before the flight, because the loss on `locincrement` is that feature's reconstruction error given all ten, not a flag for a new place. |
| R-B-L5-005, session duration anomaly | **1** | An office worker's eleven-hour day against four of eight. Her colleagues' eight-hour days measure at exactly their baseline. The service account's longest night would fire too, and is excluded because the identity context marks it a service account. |
| R-C-001, lateral movement chain | **1** | One chain, and the first chained rule here to return a row, because it reads scored events rather than notables: a fan-out rising to twenty-five new addresses -- half R-B-L3-001's floor -- then a first login from that address to a server, then a process on the server no peer had run, inside thirty minutes. The directory names the server `SRV-DB-02` and the EDR `srv-db-02`, and both pipelines render the two as one host. Seven actors each miss one step and are quiet: the process before the login, the last step at thirty-five minutes, the principal's own server, a login from another address, a flat fan-out, a server running only its routine, and a finance workstation where the rule wants a server. |
| R-C-004, staged exfiltration | **1** | One chain: a bulk export, twenty-five minutes later a fifty-times breach from the address the exporter's open session held, and twenty minutes after that a connection to a destination whose issuer nobody in the estate had seen. Four actors each do all three with one relation broken and are quiet: another principal's address, a breach after logging off, the corporate issuer, and the export last rather than first. Every step carries the session that held its address, so the chain is joined on the session, `hard:session_key`. |
| R-C-002, TLS before beaconing | **1** | One chain, and the first this rule has ever returned: it now reads the scored events rather than the two detections' notables. A settled host presents a new stack to a destination and fourteen minutes later its beacon there matures. Four hosts do both halves with one condition broken and are quiet: a beacon already running an hour before, a beacon to another address, a beacon maturing after sixty-seven minutes, and a host with five handshakes behind it. The new stack and the beacon's first flow are one connection, so the pair is joined on its Community ID, `hard:community_id`. |
| R-C-005, credential replay across the stack | **1** | One principal at two switch ports 534 km apart twenty minutes apart: the leases name the workstations behind both sign-in addresses, and the MAC bindings closed from the two sites' switches put them at headquarters and in Edinburgh. Five others are one step short and quiet -- two ports at the same site, the same journey in three hours, an address no lease names, a lease that had ended an hour before, and a refused second attempt. Nothing here rests on geolocation: both ends of the journey are ports. |
| R-C-003, physical compromise chain | **1** | One port and one device: the walk that finds a single-board computer behind a headquarters desk port, a vendor the port has never carried, on the poll that saw its link drop, and thirteen minutes later SSH from the address its lease gave it to the switches' management interface. A fleet laptop plugged in the same way, a printer that never reaches management, and a second single-board computer behind a desk switch whose link never moves are each one step short and quiet. |
| Behavior summary, per-layer scores | **9032** | One row per five-minute bin, layer, entity and lineage over the 18666 scored events, of which 1568 are the host segment's, one per workstation per hour on `morpheus:score:host`. It returned nothing until `EnvelopeStampStage` put `osi_layer` and `entity_key` on every record, and 320 until the estate pipeline rendered these events with the desk authentications beside the ports; `peak_z` is still null outside layer 5's scored week and the host segment's, because only `TC5ScoreStage` produces `max_abs_z` and it does not score the fortnight the models were trained on. |
| Chain assembly, cross-layer risk | **1** | One chain: Carol's port, in the window the tap and the desk spoof share. Every detection collects its rows into `behavior_risk` and this search sums them per lineage, once each. 15 of the 3772 chains span three layers -- a port's layer 1 samples, the layer 2 observations resolved onto it, and the authentications of the person sitting there -- and four detections' records land on them: R-D-L1-002, the tap on Carol's port, at 50, and R-D-L1-010, the same lost light read against the port's commissioning level, at 30; R-D-L2-003, the gateway claimed by two MACs, at 55 on the flooding host's port and, since the workstation on the tapped port claims the gateway too, at 55 on Carol's. Carol's chain sums to 135 and is returned, naming its three rules and the three methods the ladder took; the flooding host's chain stays at 55, under the 60 the search needs, and the other 13 carry none. The run records all three totals. `methods` reads every hop the resolver ladder took, so the 15 name `soft:directory`, `soft:dot1x` and `soft:mac_table` between them. No search head has returned the row yet: the thirteenth run was made over the events before the desk spoof. |
| Binding lookup, L2/L3 refresh | **1685** | One row per desk address and five-minute bucket its lease covers: the campaign DHCP server's acknowledgements for the eleven addresses R-C-005's sign-ins come from and the four R-C-003's intruders were given, closed by `TC2BindingStage` into leases held for the term the server granted, each naming the workstation's MAC and the switch port the relay saw. It was empty until the leases had a producer, and for a while before that was credited with 80 rows that were a MAC table under a different name. |
| Binding lookup, L2 refresh | **1249** | One row per MAC and five-minute bucket a MAC table binding covers: the estate's 124 and the campaign's two sites' 1125, the intruders' four desk ports among them. Each names the port the MAC was behind, the rung between an address's MAC and a port. |
| Binding lookup, L5/L2 refresh | **59** | One row per 802.1X identity and five-minute bucket a supplicant binding covers, from the estate's authenticator: the port an identity was authorized on, the rung from a person to a port once the estate's directory names the identity they present. |
| Binding lookup, L1 refresh | **23** | Twenty-three port intervals across thirteen ports: eight that never changed, two on each of the two ports whose optics are swapped and on the re-patched and inserted-into ports, and three on each of the restarted switch's ports, whose neighbour was unknown for one poll. The lookup keys on port and switch with no bucket, so a port's intervals collapse to one row and the latest wins -- it answers what is in a port now, not what was in it then. |
| Binding lookup, L1 history refresh | **7** | Seven rows, one per port whose binding was ever superseded: the two re-fitted optics, the re-patch, the insertion, and the restarted switch's three ports. The six that never changed are described for all time by the current-state row and cost the history nothing. The restart is a cost the history pays, recorded rather than hidden: the binding stage cannot tell a neighbour not yet relearned from a cable pulled. |
| Binding lookup, L1 history expiry | **7** | Keeps every day bucket inside the retention window and rewrites the lookup with them, and every one is inside it here: the refresh keys each row on the day its own time falls in, which the run moves to the weeks before it. Its zero, and the other expiry jobs', was a bucket counted from 1970 measured against `now()`. |
| Principal watchlist, expiry | **2** | The runner dates the events so the newest lands three to ten days before the run. R-P-L7-006's two entries are stamped with their principals' latest events, three days behind the newest in the corpus, so they are still inside their thirty days; R-P-L5-006's six are stamped more than thirty-five days behind it and are dropped. It was written as 0 until the first search-head run showed that only the undated 1970 events expire everything. Run it last: run before R-B-L7-002, it empties the list that search reads. |
| Host watchlist, expiry | **3** | R-P-L3-009's three entries are stamped with the drifting host's last three days, which are the newest events in the corpus, so all three are inside their thirty days once the runner has dated them, and the job keeps them. R-P-L6-006's three and R-P-L7-009's and R-P-L7-010's four each are stamped in the first week of the presentation and application corpora, more than fifty days behind, and R-P-L7-011's three in the creeping build server's twenty-eighth to thirtieth days, more than twenty-seven days behind; all fourteen are dropped. Run it last, as the principals' is: run before R-B-L7-004, it empties the list that search reads. |
| Binding lookup, L2/L3 expiry | **1685** | The same, for the address rung's five-minute buckets: it keeps all of them. |
| Binding lookup, L2 expiry | **1249** | The same, for the MAC rung. |
| Binding lookup, L5/L2 expiry | **59** | The same, for the 802.1X identity rung. |
| Binding health, unresolved rate | **1** | The ARP stream, the one class whose records carry a resolution outcome: 180 unresolved of 1,220 (0.148), under the 0.2 that marks a class degraded. It read five while the wire sent null fields: four classes carry `resolution_method` with no value, `resolution_method=*` matched the nulls, and the search reported them as resolving with nothing unresolved. An operational metric; the value matters, not whether it fired. |
| Pipeline health, refusal rate | **1** | The DMZ poller: three polls reported 4500 ms out on an undisciplined clock, all three quarantined, none delivered -- a refusal rate of 1.0. Every other collector delivered everything. An operational metric. |
| Collector health | **1** | The ARP sensor, whose sequence skips the three observations it never delivered. No collector repeats a number, every offset is within twenty milliseconds, and every record reached the pipeline two seconds after the sensor saw it. Empty is the healthy answer. |
| Pipeline health, stage counts | **5** | What the stages could not do, by collector and class: the host frame's 56 and the sign-ins' 20 rows a fallback model scored, the RADIUS authorizations' two unpaired exchanges, the RADIUS accounting feed's two unpaired session records and one abandoned session, and the identity provider's one session record out of order. A count no stage of a class writes is null and stays out of the sum. An operational metric. |
| Late arrivals by class | **1** | The layer 3 corpus's one late flow, routed to `morpheus:late` and so read by no detection. |
| Platform baseline step | **1** | The SaaS corpus's second tenant, whose sync doubled for its last two days and held there. The identity provider's one-hour burst to twice its peak does not hold into the next window and is not a step. |
| R-P-L5-006, drift trajectory | **6** | Two principals, neither of them behaviour, explained in `expected_results.json`: the two whose week repeats their fortnight climb on its last three days by hundredths a day, as the cadence features keep moving after their models were trained, and the rule measures that against a day-to-day spread that is nearly zero. The principals whose week does depart all cross the mean ceiling and leave. Watchlist, never a page: every firing is written to `principal_watchlist`. |

**One of the eighty-six should return nothing.** That is the point of writing them down. An empty result is
this app's characteristic failure, and without a list saying which emptiness is correct, a deployment cannot
tell a rule that is working from a rule that is broken. The ratio has moved both ways, which is what makes it
worth stating: it improved as layers 3, 4, 5, 6 and 7 gained producers, and went the other way when the L2/L3 refresh
was found to have been empty all along under a note that credited it with 80 rows -- it left the list again
when the DHCP leases gained a producer, and the expiry jobs left with it once each refresh keyed its rows on
their own time rather than on a bucket counted from 1970 -- and again when R-P-L3-005
turned out to have been empty because its own SPL could not fire. R-B-L5-001 and R-B-L5-002 left the list when
each principal gained a learned model of their own, which is what their gate had been waiting for. Chain assembly is the one worth following,
because it was empty for four different reasons in turn: a missing `osi_layer`, then lineage that never left one
layer, then a risk sum nothing wrote, then a risk sum that was written and fell five points short on the one
three-layer chain a detection accused. Each fix made the next blocker visible, which is what an expectation file is
for, and the last of them was the corpus: no planted actor was ever accused twice on one chain, so the search was
validated only as a correct empty until the desk spoof gave it the man-in-the-middle it exists to assemble.

## The watchlist the predictive searches write

R-P-L5-006 and R-P-L7-006 write the `principal_watchlist` KV Store lookup as they run, and R-B-L7-002 reads
it. Run the predictive searches first, then

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

The hosts have a list of their own. R-P-L3-009, R-P-L6-006, R-P-L7-009, R-P-L7-010 and R-P-L7-011 write the
`host_watchlist` lookup as they run, keyed on the host, the rule and the day, and R-B-L7-004 reads it by the host its rows name to raise a watched host's
notables by 15.

```
| inputlookup host_watchlist | stats count BY rule_id reason
```

should return five rows: `R-P-L3-009` with `host-drift-trajectory` and 3 entries, one per host-day it fired on;
`R-P-L6-006` with `stack-trajectory` and 3 entries, the creeping host's last three days;
`R-P-L7-009` with `subdomain-trajectory` and 4 entries, the slow tunnel's last four;
`R-P-L7-010` with `path-trajectory` and 4 entries, the slow enumerator's last four; and
`R-P-L7-011` with `process-breadth-trajectory` and 3 entries, the creeping build server's. One of the hosts R-B-L7-004
reports is on it: the build server whose `curl` the rule raises from 60 to 75, nine days after the list took it for
widening. The other machines on the list share nothing with the endpoint corpus's or the campaign's.
Its expiry job keeps R-P-L3-009's three entries, because the drifting host's last three days are the newest events in
the corpus, and drops the other fourteen: eleven whose days sit in the first week of the presentation and application
corpora, more than fifty days behind, and the build server's three, more than twenty-seven.

## Where the sample events come from

[`make_sample_events.py`](./make_sample_events.py) runs the same `run_pipeline` the determinism tests call and
puts the output through the same `SiemWireStage` a deployment would put before its sink. Layers 1, 2 and 5 come
from the estate pipeline rather than the telemetry one, because a chain is decided by which classes were sealed
together: the same events rendered from the telemetry pipeline carry chains that stop at two layers. Layers 3
4, 6 and 7 come from their own corpora and their own pipelines, and share an entity with neither the estate
nor each other -- an address is not a port, a conversation is not an address, and the host a handshake or a query
was made from is not any of them. The host segment's rows come from the host score corpus, one per workstation per
hour, scored against the committed peer-group models, on `morpheus:score:host`. Each scored corpus's sealed output also passes through `PipelineHealthStage`, as a
deployment's health branch would, onto `morpheus:health`; a row its sealer marked late goes to `morpheus:late`
instead of its score sourcetype; and the records `EnvelopeValidateStage` quarantined go to `morpheus:quarantine`.
They are checked in so
that a change to what a SIEM would receive shows up in a pull request rather than only on a search head.

Regenerate after changing the corpus or a stage, and review the diff:

```bash
python examples/splunk_lineage_app/validate/make_sample_events.py
```

## What this does not establish

That a particular Splunk version parses the app the way `props.conf` says. This runs one version, in one
container, with everything on one instance. A real estate parses on indexers or heavy forwarders, and the
`KV_MODE` and acceleration choices in `props.conf` are estate-wide decisions this package does not make.
