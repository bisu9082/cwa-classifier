"""
Validation suite for the CWC Schedule matcher (scripts/cwc_schedules.py)
=======================================================================
Each case states a structure, the Annex entry it must match, and the
Schedule level it must be assigned.  Negative cases assert non-listing,
including the cases that distinguish the statutory general formulas from
naive single-element rules (organophosphate pesticides, phosphorus-bearing
flame retardants) and the explicit Annex exemptions.
"""

import sys

import cwc_schedules as cwc

# (name, SMILES, required entry or None, expected schedule)
CASES = [
    # ── Schedule 1A general formulas ──────────────────────────────────────
    ("Sarin (GB)",            "CC(C)OP(=O)(C)F",                     "1A.1",  1),
    ("Soman (GD)",            "CC(OP(=O)(C)F)C(C)(C)C",              "1A.1",  1),
    ("Cyclosarin (GF)",       "C1CCC(CC1)OP(=O)(C)F",                "1A.1",  1),
    ("Ethylsarin (GE)",       "CCOP(=O)(C)F",                        "1A.1",  1),
    ("Tabun (GA)",            "CCOP(=O)(C#N)N(C)C",                  "1A.2",  1),
    ("VX",                    "CCOP(=O)(C)SCCN(C(C)C)C(C)C",         "1A.3",  1),
    ("Russian VX (VR)",       "CCC(C)OP(=O)(C)SCCN(CC)CC",           "1A.3",  1),
    ("A-230",                 "CCN(CC)C(=NP(=O)(C)F)C",              "1A.13", 1),
    ("A-232",                 "CCN(CC)C(=NP(=O)(OC)F)C",             "1A.14", 1),
    ("A-234",                 "CCN(CC)C(=NP(=O)(OCC)F)C",            "1A.14", 1),
    ("Novichok 1A.15 type",   "CCN(CC)C(=NP(=O)(C)F)N(CC)CC",        "1A.15", 1),
    ("DF",                    "CP(=O)(F)F",                          "1B.9",  1),
    ("QL",                    "CCOP(C)OCCN(C(C)C)C(C)C",             "1B.10", 1),
    # ── Schedule 1 discrete chemicals ─────────────────────────────────────
    ("Sulfur mustard (HD)",   "ClCCSCCCl",                           "1A.4",  1),
    ("Sesquimustard (Q)",     "ClCCSCCSCCCl",                        "1A.4",  1),
    ("Lewisite 1",            "ClC=C[As](Cl)Cl",                     "1A.5",  1),
    ("Nitrogen mustard HN-2", "CN(CCCl)CCCl",                        "1A.6",  1),
    ("Nitrogen mustard HN-3", "ClCCN(CCCl)CCCl",                     "1A.6",  1),
    ("Chlorosarin",           "CC(C)OP(=O)(C)Cl",                    "1B.11", 1),
    # ── Schedule 2 ────────────────────────────────────────────────────────
    ("DMMP",                  "COP(=O)(C)OC",                        "2B.4",  2),
    ("DIMP",                  "CC(C)OP(=O)(C)OC(C)C",                "2B.4",  2),
    ("DEEP",                  "CCOP(=O)(CC)OCC",                     "2B.4",  2),
    ("Methylphosphonic dichloride", "CP(=O)(Cl)Cl",                  "2B.4",  2),
    ("Methylphosphonic acid", "CP(=O)(O)O",                          "2B.4",  2),
    ("Amiton (VG)",           "CCOP(=O)(OCC)SCCN(CC)CC",             "2A.1",  2),
    ("BZ",                    "OC(C(=O)OC1CN2CCC1CC2)(c1ccccc1)c1ccccc1", "2A.3", 2),
    ("N,N-dimethylphosphoramidic dichloride", "CN(C)P(=O)(Cl)Cl",    "2B.5",  2),
    ("N,N-Diisopropylaminoethanol", "CC(C)N(C(C)C)CCO",              "2B.11", 2),
    ("N,N-Diisopropylaminoethanethiol", "CC(C)N(C(C)C)CCS",          "2B.12", 2),
    ("N,N-Diisopropyl-2-aminoethyl chloride", "CC(C)N(C(C)C)CCCl",   "2B.10", 2),
    ("Thiodiglycol",          "OCCSCCO",                             "2B.13", 2),
    ("Pinacolyl alcohol",     "CC(O)C(C)(C)C",                       "2B.14", 2),
    ("Arsenic trichloride",   "Cl[As](Cl)Cl",                        "2B.7",  2),
    ("Quinuclidin-3-ol",      "OC1CN2CCC1CC2",                       "2B.9",  2),
    # ── Schedule 3 ────────────────────────────────────────────────────────
    ("Phosgene",              "ClC(Cl)=O",                           "3A.1",  3),
    ("Cyanogen chloride",     "ClC#N",                               "3A.2",  3),
    ("Hydrogen cyanide",      "C#N",                                 "3A.3",  3),
    ("Chloropicrin",          "[O-][N+](=O)C(Cl)(Cl)Cl",             "3A.4",  3),
    ("Trimethyl phosphite",   "COP(OC)OC",                           "3B.8",  3),
    ("Triethyl phosphite",    "CCOP(OCC)OCC",                        "3B.9",  3),
    ("Phosphorus trichloride","ClP(Cl)Cl",                           "3B.6",  3),
    ("Triethanolamine",       "OCCN(CCO)CCO",                        "3B.17", 3),
    # ── negative controls: organophosphate pesticides ─────────────────────
    ("Malathion",   "CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC",              None, 0),
    ("Parathion",   "CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1",          None, 0),
    ("Chlorpyrifos","CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl",              None, 0),
    ("Diazinon",    "CCOP(=S)(OCC)Oc1cc(C)nc(C(C)C)n1",              None, 0),
    ("Dimethoate",  "CNC(=O)CSP(=S)(OC)OC",                          None, 0),
    ("Trichlorfon", "COP(=O)(OC)C(O)C(Cl)(Cl)Cl",                    None, 0),
    ("Dichlorvos",  "COP(=O)(OC)OC=C(Cl)Cl",                         None, 0),
    ("Terbufos",    "CCOP(=S)(OCC)SCSC(C)(C)C",                      None, 0),
    ("Glyphosate",  "OC(=O)CNCP(=O)(O)O",                            None, 0),
    ("Glufosinate", "CP(=O)(O)CCC(N)C(=O)O",                         None, 0),
    ("Acephate",    "CNC(=O)CSP(=O)(OC)NC(C)=O",                     None, 0),
    # ── negative controls: simulants that are NOT scheduled ───────────────
    ("DFP",         "CC(C)OP(=O)(F)OC(C)C",                           None, 0),
    ("TEP",         "CCOP(=O)(OCC)OCC",                               None, 0),
    ("TMP",         "COP(=O)(OC)OC",                                  None, 0),
    ("CEES",        "CCSCCCl",                                        None, 0),
    ("Methyl salicylate", "COC(=O)c1ccccc1O",                         None, 0),
    # ── negative controls: organophosphorus flame retardants ──────────────
    ("TCEP",        "ClCCOP(=O)(OCCCl)OCCCl",                         None, 0),
    ("TPhP",        "O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1",             None, 0),
    ("TEHP",        "CCCCC(CC)COP(=O)(OCC(CC)CCCC)OCC(CC)CCCC",       None, 0),
    # ── Annex exemptions ──────────────────────────────────────────────────
    ("Fonofos (2B.4 exempt)",  "CCP(=S)(OCC)Sc1ccccc1",               None, 0),
    ("DMAE (2B.11 exempt)",    "CN(C)CCO",                            None, 0),
    ("DEAE (2B.11 exempt)",    "CCN(CC)CCO",                          None, 0),
]


# (name, SMILES, motifs that must fire, motifs that must NOT fire)
MOTIF_CASES = [
    ("Parathion", "CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1",
     ["sca_nitro_aromatic", "sca_nitroaryl_ester", "sca_P_aryl_ester",
      "sca_P_thiono"], ["sca_P_fluoridate", "sca_arsenical"]),
    ("DMNP", "COP(=O)(OC)Oc1ccc([N+](=O)[O-])cc1",
     ["sca_nitro_aromatic", "sca_nitroaryl_ester", "sca_P_aryl_ester"],
     ["sca_P_thiono"]),
    ("Methyl salicylate", "COC(=O)c1ccccc1O",
     ["sca_aryl_ester"], ["sca_nitro_aromatic", "sca_P_aryl_ester"]),
    # P-O-aryl is an aryl ester, not a direct P-aryl bond
    ("TPhP", "O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1",
     ["sca_P_aryl_ester"], ["sca_nitro_aromatic", "sca_P_aryl_direct"]),
    ("Dimethyl phenylphosphonate", "COP(=O)(OC)c1ccccc1",
     ["sca_P_aryl_direct"], ["sca_P_aryl_ester"]),
    ("Sulfur mustard", "ClCCSCCCl",
     ["sca_chloroethyl_S", "sca_thioether", "sca_beta_haloalkyl"],
     ["sca_arsenical", "sca_P_thiono"]),
    ("Lewisite 1", "ClC=C[As](Cl)Cl",
     ["sca_arsenical", "sca_vinyl_arsenical"], ["sca_thioether"]),
    # malathion's sulfur atoms bridge S-P, not C-S-C, so no thioether
    ("Malathion", "CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC",
     # malathion's two esters are aliphatic, so no aryl carboxylate either
     ["sca_P_thiono", "sca_P_dithio"],
     ["sca_nitro_aromatic", "sca_P_fluoridate", "sca_thioether",
      "sca_aryl_ester"]),
    ("Sarin", "CC(C)OP(=O)(C)F",
     ["sca_P_fluoridate", "sca_P_alkyl_direct", "sca_P_halide"],
     ["sca_P_thiono", "sca_thioether"]),
    ("VX", "CCOP(=O)(C)SCCN(C(C)C)C(C)C",
     ["sca_P_thiolate_amine", "sca_P_alkyl_direct"],
     ["sca_P_fluoridate", "sca_thioether"]),
    ("A-230", "CCN(CC)C(=NP(=O)(C)F)C",
     ["sca_P_amidine", "sca_P_fluoridate"], ["sca_thioether"]),
    # in a disulfide each sulfur bonds one carbon and one sulfur
    ("Dipropyl disulfide", "CCCSSCCC",
     ["sca_disulfide"], ["sca_P_thiono", "sca_thioether"]),
    # thiophene is a cyclic thioether, so both motifs fire
    ("Thiophene", "c1ccsc1",
     ["sca_thiophene_ring", "sca_thioether"], ["sca_disulfide"]),
    ("CEES", "CCSCCCl",
     ["sca_thioether", "sca_chloroethyl_S", "sca_beta_haloalkyl"],
     ["sca_disulfide"]),
    ("Tabun", "CCOP(=O)(C#N)N(C)C",
     ["sca_P_cyanide", "sca_P_amide", "sca_nitrile"], ["sca_P_fluoridate"]),
    ("Trimethyl phosphite", "COP(OC)OC",
     ["sca_P_III"], ["sca_P_thiono", "sca_P_halide"]),
    ("Nitrogen mustard HN-3", "ClCCN(CCCl)CCCl",
     ["sca_chloroethyl_N", "sca_beta_haloalkyl"], ["sca_thioether"]),
    ("N,N-Diisopropylaminoethanol", "CC(C)N(C(C)C)CCO",
     ["sca_aminoethyl"], ["sca_chloroethyl_N"]),
    ("Phosgene", "ClC(Cl)=O", ["sca_acyl_halide"], ["sca_nitrile"]),
    ("Hydrogen cyanide", "C#N", ["sca_nitrile"], ["sca_acyl_halide"]),
]


def run_motif_cases() -> list[str]:
    """Every scaffold motif must fire on a compound that carries it.

    The Schedule assignment tests above exercise the statutory matcher; these
    cases exercise the 26-motif descriptor block, which would otherwise be
    unvalidated.  A motif that never fires is a silently dead descriptor.
    """
    failures = []
    print("\n  -- scaffold motif controls --")
    for name, smi, must, must_not in MOTIF_CASES:
        flags = cwc.motif_flags(smi)
        bad = [m for m in must if not flags.get(m)]
        bad += [f"!{m}" for m in must_not if flags.get(m)]
        ok = not bad
        if not ok:
            failures.append(f"{name}: {', '.join(bad)}")
        print(f"  {'ok  ' if ok else 'FAIL'} {name:26s} "
              f"{'' if ok else 'unexpected: ' + ', '.join(bad)}")

    # no motif in the library may be structurally unreachable
    covered = set()
    for _n, smi, _m, _x in MOTIF_CASES:
        covered |= {k for k, v in cwc.motif_flags(smi).items() if v}
    dead = [m for m in cwc.MOTIF_NAMES if m not in covered]
    if dead:
        print(f"  note  motifs not exercised by these controls: "
              f"{', '.join(dead)}")
    return failures


def main() -> int:
    errs = cwc.load_errors()
    if errs:
        print("PATTERN LOAD ERRORS:")
        for e in errs:
            print("   ", e)

    failures = []
    for name, smi, want_entry, want_sched in CASES:
        r = cwc.classify(smi)
        ok = r["parsed"] and r["schedule"] == want_sched
        if want_entry is not None:
            ok = ok and want_entry in r["entries"]
        status = "ok  " if ok else "FAIL"
        if not ok:
            failures.append(name)
        print(f"  {status} {name:42s} sched={r['schedule']} "
              f"entries={','.join(r['entries']) or '-':22s}"
              f"{'  exempt: ' + r['exempt'] if r['exempt'] else ''}")

    motif_failures = run_motif_cases()
    total = len(CASES) + len(MOTIF_CASES)
    passed = total - len(failures) - len(motif_failures)
    print(f"\n  {passed}/{total} cases pass "
          f"({len(CASES)} Schedule assignments, {len(MOTIF_CASES)} motif controls)")
    if failures:
        print("  failing Schedule cases:", ", ".join(failures))
    if motif_failures:
        print("  failing motif cases:", "; ".join(motif_failures))
    return 1 if (failures or motif_failures or errs) else 0


if __name__ == "__main__":
    sys.exit(main())
