# Budget and Procurement

Plan v4 authorizes Engine v0 without the complete hardware fleet. Procurement is
tied to replacing high-impact assumptions, not to keeping implementation moving.

## Cost posture

| Category | Engine v0 | G2/G3 | Rule |
|---|---|---|---|
| People | Small cross-functional runtime/distributed team | Add kernel/validation coverage as hardware arrives | Do not wait for a full future team to implement contracts |
| Compute | Existing CPU/developer machines for simulation and loopback | Schedule/acquire exact backend devices | Each spend closes named SA-* assumptions |
| Fabric | Loopback plus injected link profiles | Buy/borrow the next useful 10/25/100 GbE or direct link | Scenario sensitivity precedes purchase |
| CI | CPU T0/T1 on every change | T2/T3 lab lanes when available | Physical jobs report hardware/build identity |

The later business metric remains cost per token/capacity versus a defined
datacenter baseline. It is not calculated until the capacity target is bound.

## Hardware bundles

| Bundle | Contents | Needed by | Status |
|---|---|---|---|
| `engine-v0` | CPU/local processes for reference/simulated backends and loopback TCP | Phase 0.5 | Available |
| `desktop-minimal` | Access to one accepted Linux accelerator and one Apple node, initially usable independently | G2 T2/T3 | Partially available; exact inventory open |
| `fabric-step` | Link selected from scenario sensitivity | G2/G3 | Open; no premature 100 GbE assumption |
| `lab-reference` | Exact frontier-capacity target fleet with controlled thermals | G3/G4 | Deferred until G2 and target selection |

## Actions

- [x] Authorize Engine v0 on reference/simulated backends.
- [ ] Inventory existing hardware and record exact available test windows.
- [ ] Run the full link/compute scenario matrix before buying fabric.
- [ ] Select the first purchase/borrow action by the SA-* assumption it closes.
- [ ] Fill exact G2 physical fleet fields in `v0-target-contract.md` as access is
      confirmed.
- [ ] Record a negative hardware list after backend tests fail or pass with limits.

## Rule

No broad frontier-fleet purchase is committed before Engine v0 scenario results
and a G2 validation plan identify its decision value. Long-lead exceptions require
a Sponsor decision record.
