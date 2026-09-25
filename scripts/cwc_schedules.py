"""
CWC Annex on Chemicals: machine-readable Schedule matcher
=========================================================
Implements the structural classification rules of the Chemical Weapons
Convention (CWC) Annex on Chemicals as executable substructure queries, so
that every compound in a corpus can be assigned its actual Schedule status
instead of being classified by single-element heuristics.

Two complementary matching modes are used, mirroring the way the Annex is
written:

*   **General-formula entries.**  Several Annex entries are written as a
    parent scaffold plus rigid substituent constraints (e.g. Schedule 1A.1,
    "O-Alkyl (<=C10, incl. cycloalkyl) alkyl (Me, Et, n-Pr or i-Pr)-
    phosphonofluoridates").  These are encoded as an RDKit SMARTS query for
    the parent scaffold, followed by a programmatic check of every
    substituent constraint the SMARTS cannot express - which alkyl groups
    are admissible on phosphorus, carbon-chain length ceilings, and the
    requirement that no further carbon atoms are bonded to phosphorus.

*   **Discretely listed chemicals.**  Other entries name individual
    chemicals.  These are matched on the connectivity block of the standard
    InChIKey, making the match insensitive to stereochemistry, salt form and
    protonation state.

Explicit Annex exemptions are honoured (Fonofos under Schedule 2B.4;
N,N-dimethyl- and N,N-diethylaminoethanol under Schedule 2B.11), and
Schedule 1 membership takes precedence over the Schedule 2B.4 general
formula, as the Annex requires.

Out of scope.  The biological-toxin entries (Schedule 1A.7 saxitoxin,
1A.8 ricin) and the Schedule 1A.16 carbamate quaternary/bisquaternary
entries are not encoded: they are not organophosphorus small molecules and
fall outside the chemical space of this benchmark.  Their omission is
recorded by `UNENCODED_ENTRIES` so that coverage is auditable.

Reference: Organisation for the Prohibition of Chemical Weapons, Annex on
Chemicals to the Chemical Weapons Convention, Schedules 1-3, as amended by
decisions C-24/DEC.4 and C-24/DEC.5 (entries 1A.13-1A.16).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

UNENCODED_ENTRIES = ["1A.7 (saxitoxin)", "1A.8 (ricin)", "1A.16 (carbamates)"]

# Alkyl groups admissible on phosphorus in the Schedule general formulas
P_ALKYLS = {"Me", "Et", "nPr", "iPr"}


# ══════════════════════════════════════════════════════════════════════════
#  substituent analysis helpers
# ══════════════════════════════════════════════════════════════════════════
def _subtree(mol, start: int, blocked: int) -> list[int]:
    """Atom indices reachable from `start` without passing through `blocked`."""
    seen, stack = {start}, [start]
    while stack:
        a = mol.GetAtomWithIdx(stack.pop())
        for nb in a.GetNeighbors():
            i = nb.GetIdx()
            if i != blocked and i not in seen:
                seen.add(i)
                stack.append(i)
    return sorted(seen)


def _is_hydrocarbyl(mol, atoms: list[int]) -> bool:
    """True if the substituent consists solely of carbon (and implicit H)."""
    return all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in atoms)


def carbon_count(mol, start: int, blocked: int) -> int | None:
    """Number of carbons in a purely hydrocarbyl substituent, else None."""
    atoms = _subtree(mol, start, blocked)
    if not _is_hydrocarbyl(mol, atoms):
        return None
    return len(atoms)


def alkyl_kind(mol, c_idx: int, p_idx: int) -> str | None:
    """Classify a carbon substituent as Me / Et / nPr / iPr, else None.

    The group must be a saturated, acyclic, unsubstituted alkyl of at most
    three carbons; anything else (including cyclopropyl, vinyl, or a carbon
    bearing a heteroatom) returns None.
    """
    atom = mol.GetAtomWithIdx(c_idx)
    if atom.GetAtomicNum() != 6:
        return None
    atoms = _subtree(mol, c_idx, p_idx)
    if not _is_hydrocarbyl(mol, atoms):
        return None
    for i in atoms:
        a = mol.GetAtomWithIdx(i)
        if a.GetIsAromatic() or a.IsInRing():
            return None
        if a.GetHybridization() != Chem.HybridizationType.SP3:
            return None
    n = len(atoms)
    if n == 1:
        return "Me"
    if n == 2:
        return "Et"
    if n == 3:
        # i-Pr: the attachment carbon carries both remaining carbons
        n_on_attach = sum(
            1 for nb in atom.GetNeighbors()
            if nb.GetAtomicNum() == 6 and nb.GetIdx() != p_idx
        )
        return "iPr" if n_on_attach == 2 else "nPr"
    return None


def o_alkyl_ok(mol, o_idx: int, p_idx: int, allow_h: bool = False,
               max_c: int = 10) -> bool:
    """Check an O-substituent against 'O-Alkyl (<=C10, incl. cycloalkyl)'.

    Cycloalkyl is admissible (hence rings are permitted), but the group must
    be carbon-only and saturated; aromatic esters are not alkyl esters.  With
    `allow_h`, a free P-OH (the 'H or <=C10' wording) also satisfies the
    constraint.
    """
    o = mol.GetAtomWithIdx(o_idx)
    heavy = [nb.GetIdx() for nb in o.GetNeighbors() if nb.GetIdx() != p_idx]
    if not heavy:
        return allow_h                      # P-OH
    if len(heavy) > 1:
        return False
    c = heavy[0]
    atoms = _subtree(mol, c, o_idx)
    if not _is_hydrocarbyl(mol, atoms) or len(atoms) > max_c:
        return False
    return not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)


def dialkylamino_ok(mol, n_idx: int, exclude: int | None = None) -> bool:
    """Check an amine nitrogen against 'N,N-dialkyl (Me, Et, n-Pr or i-Pr)'."""
    n = mol.GetAtomWithIdx(n_idx)
    cs = [nb.GetIdx() for nb in n.GetNeighbors()
          if nb.GetAtomicNum() == 6 and nb.GetIdx() != exclude]
    alkyls = [alkyl_kind(mol, c, n_idx) for c in cs]
    alkyls = [a for a in alkyls if a in P_ALKYLS]
    return len(alkyls) >= 2


def dialkylamino_c10_ok(mol, n_idx: int, exclude: int | None = None) -> bool:
    """Check an amine nitrogen against 'dialkyl (<=C10, incl. cycloalkyl)amino'.

    Entries 1A.13-1A.15 place a C10 ceiling on the amino substituents rather
    than restricting them to methyl/ethyl/propyl, so the admissible groups
    are any two saturated carbon-only substituents of at most ten carbons.
    """
    n = mol.GetAtomWithIdx(n_idx)
    cs = [nb.GetIdx() for nb in n.GetNeighbors()
          if nb.GetAtomicNum() == 6 and nb.GetIdx() != exclude]
    ok = 0
    for c in cs:
        atoms = _subtree(mol, c, n_idx)
        if (_is_hydrocarbyl(mol, atoms) and len(atoms) <= 10
                and not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)):
            ok += 1
    return ok >= 2


def alkylidene_ok(mol, c_idx: int, blocked: tuple[int, ...]) -> bool:
    """Check the alkylidene carbon substituent of entries 1A.13-1A.15.

    The wording is 'alkylidene (H or <=C10, incl. cycloalkyl)', so the carbon
    may carry hydrogen only, or a saturated carbon-only group of at most ten
    carbons.
    """
    c = mol.GetAtomWithIdx(c_idx)
    subs = [nb.GetIdx() for nb in c.GetNeighbors() if nb.GetIdx() not in blocked]
    for s in subs:
        a = mol.GetAtomWithIdx(s)
        if a.GetAtomicNum() != 6:
            return False
        atoms = _subtree(mol, s, c_idx)
        if (not _is_hydrocarbyl(mol, atoms) or len(atoms) > 10
                or any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)):
            return False
    return True


def p_bonded_carbons(mol, p_idx: int) -> list[int]:
    """Carbon atoms bonded directly to a phosphorus atom."""
    return [nb.GetIdx() for nb in mol.GetAtomWithIdx(p_idx).GetNeighbors()
            if nb.GetAtomicNum() == 6]


# ══════════════════════════════════════════════════════════════════════════
#  general-formula entries
# ══════════════════════════════════════════════════════════════════════════
def _p_atoms(mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 15]


def _nbrs(mol, idx, z, order=None):
    out = []
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        nb = b.GetOtherAtom(mol.GetAtomWithIdx(idx))
        if nb.GetAtomicNum() == z and (order is None or b.GetBondType() == order):
            out.append(nb.GetIdx())
    return out


def _has_px_o(mol, p) -> bool:
    """Phosphoryl or thiophosphoryl P=O / P=S."""
    return bool(_nbrs(mol, p, 8, Chem.BondType.DOUBLE)
                or _nbrs(mol, p, 16, Chem.BondType.DOUBLE))


def m_1A1(mol) -> bool:
    """1A.1 O-Alkyl (<=C10, incl. cycloalkyl) alkyl(Me,Et,n-Pr,i-Pr)
    phosphonofluoridates."""
    for p in _p_atoms(mol):
        if not _nbrs(mol, p, 8, Chem.BondType.DOUBLE):
            continue
        f = _nbrs(mol, p, 9, Chem.BondType.SINGLE)
        cs = p_bonded_carbons(mol, p)
        os_ = _nbrs(mol, p, 8, Chem.BondType.SINGLE)
        if len(f) == 1 and len(cs) == 1 and len(os_) == 1:
            if alkyl_kind(mol, cs[0], p) in P_ALKYLS and o_alkyl_ok(mol, os_[0], p):
                return True
    return False


def m_1A2(mol) -> bool:
    """1A.2 O-Alkyl (<=C10) N,N-dialkyl(Me,Et,n-Pr,i-Pr)
    phosphoramidocyanidates."""
    q = Chem.MolFromSmarts("[#15X4](=[OX1])([CX2]#[NX1])([OX2])[NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, _, _, o, n = match
        if o_alkyl_ok(mol, o, p) and dialkylamino_ok(mol, n, exclude=p):
            return True
    return False


def m_1A3(mol) -> bool:
    """1A.3 O-Alkyl (H or <=C10) S-2-dialkylaminoethyl
    alkyl(Me,Et,n-Pr,i-Pr) phosphonothiolates."""
    q = Chem.MolFromSmarts(
        "[#15X4](=[OX1])([CX4])([OX2])[SX2][CH2][CH2][NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, c, o, _s, _c1, _c2, n = match
        if (alkyl_kind(mol, c, p) in P_ALKYLS
                and o_alkyl_ok(mol, o, p, allow_h=True)
                and dialkylamino_ok(mol, n)):
            return True
    return False


def m_1A13(mol) -> bool:
    """1A.13 P-alkyl (H or <=C10, incl. cycloalkyl)
    N-(1-(dialkyl(<=C10)amino)alkylidene(H or <=C10))
    phosphonamidic fluorides."""
    q = Chem.MolFromSmarts("[#15X4](=[OX1])(F)[NX2]=[CX3][NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, _f, n1, c, n2 = match
        cs = p_bonded_carbons(mol, p)
        if len(cs) != 1:
            continue
        nc = carbon_count(mol, cs[0], p)
        if (nc is not None and nc <= 10
                and dialkylamino_c10_ok(mol, n2)
                and alkylidene_ok(mol, c, (n1, n2))):
            return True
    return False


def m_1A14(mol) -> bool:
    """1A.14 O-alkyl (H or <=C10, incl. cycloalkyl)
    N-(1-(dialkyl(<=C10)amino)alkylidene(H or <=C10))
    phosphoramidofluoridates."""
    q = Chem.MolFromSmarts("[#15X4](=[OX1])(F)([OX2])[NX2]=[CX3][NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, _f, o, n1, c, n2 = match
        if (not p_bonded_carbons(mol, p)
                and o_alkyl_ok(mol, o, p, allow_h=True)
                and dialkylamino_c10_ok(mol, n2)
                and alkylidene_ok(mol, c, (n1, n2))):
            return True
    return False


def m_1A15(mol) -> bool:
    """1A.15 Methyl-(bis(dialkylamino)methylene)phosphonamidofluoridate."""
    q = Chem.MolFromSmarts("[#15X4](=[OX1])(F)[NX2]=[CX3]([NX3])[NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, _f, _n1, _c, n2, n3 = match
        cs = p_bonded_carbons(mol, p)
        if len(cs) == 1 and alkyl_kind(mol, cs[0], p) in P_ALKYLS:
            if dialkylamino_c10_ok(mol, n2) and dialkylamino_c10_ok(mol, n3):
                return True
    return False


def m_1B9(mol) -> bool:
    """1B.9 Alkyl(Me,Et,n-Pr,i-Pr) phosphonyldifluorides."""
    for p in _p_atoms(mol):
        if not _nbrs(mol, p, 8, Chem.BondType.DOUBLE):
            continue
        f = _nbrs(mol, p, 9, Chem.BondType.SINGLE)
        cs = p_bonded_carbons(mol, p)
        if len(f) == 2 and len(cs) == 1 and alkyl_kind(mol, cs[0], p) in P_ALKYLS:
            return True
    return False


def m_1B10(mol) -> bool:
    """1B.10 O-Alkyl (H or <=C10) O-2-dialkylaminoethyl
    alkyl(Me,Et,n-Pr,i-Pr) phosphonites."""
    q = Chem.MolFromSmarts("[#15X3]([CX4])([OX2])[OX2][CH2][CH2][NX3]")
    for match in mol.GetSubstructMatches(q):
        p, c, o1, _o2, _c1, _c2, n = match
        if (alkyl_kind(mol, c, p) in P_ALKYLS
                and o_alkyl_ok(mol, o1, p, allow_h=True)
                and dialkylamino_ok(mol, n)):
            return True
    return False


def m_2B4(mol) -> bool:
    """2B.4 Chemicals, except those listed in Schedule 1, containing a
    phosphorus atom to which is bonded one methyl, ethyl or propyl (normal
    or iso) group but not further carbon atoms.

    Schedule 1 precedence and the Fonofos exemption are applied by the
    caller (`classify`), not here.
    """
    for p in _p_atoms(mol):
        cs = p_bonded_carbons(mol, p)
        if len(cs) == 1 and alkyl_kind(mol, cs[0], p) in P_ALKYLS:
            return True
    return False


def m_2B5(mol) -> bool:
    """2B.5 N,N-Dialkyl(Me,Et,n-Pr,i-Pr) phosphoramidic dihalides."""
    q = Chem.MolFromSmarts("[#15X4](=[OX1])([F,Cl,Br,I])([F,Cl,Br,I])[NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, _x1, _x2, n = match
        if dialkylamino_ok(mol, n, exclude=p):
            return True
    return False


def m_2B6(mol) -> bool:
    """2B.6 Dialkyl(Me,Et,n-Pr,i-Pr) N,N-dialkyl(Me,Et,n-Pr,i-Pr)-
    phosphoramidates."""
    q = Chem.MolFromSmarts("[#15X4](=[OX1])([OX2])([OX2])[NX3]")
    for match in mol.GetSubstructMatches(q):
        p, _, o1, o2, n = match
        ok = True
        for o in (o1, o2):
            heavy = [nb.GetIdx() for nb in mol.GetAtomWithIdx(o).GetNeighbors()
                     if nb.GetIdx() != p]
            if len(heavy) != 1 or alkyl_kind(mol, heavy[0], o) not in P_ALKYLS:
                ok = False
        if ok and dialkylamino_ok(mol, n, exclude=p):
            return True
    return False


def _aminoethyl_x(mol, smarts: str) -> bool:
    q = Chem.MolFromSmarts(smarts)
    for match in mol.GetSubstructMatches(q):
        n = match[0]
        if dialkylamino_ok(mol, n):
            return True
    return False


def m_2B10(mol) -> bool:
    """2B.10 N,N-Dialkyl(Me,Et,n-Pr,i-Pr) aminoethyl-2-chlorides."""
    return _aminoethyl_x(mol, "[NX3]([CX4])([CX4])[CH2][CH2][Cl]")


def m_2B11(mol) -> bool:
    """2B.11 N,N-Dialkyl(Me,Et,n-Pr,i-Pr) aminoethane-2-ols."""
    return _aminoethyl_x(mol, "[NX3]([CX4])([CX4])[CH2][CH2][OX2H1]")


def m_2B12(mol) -> bool:
    """2B.12 N,N-Dialkyl(Me,Et,n-Pr,i-Pr) aminoethane-2-thiols."""
    return _aminoethyl_x(mol, "[NX3]([CX4])([CX4])[CH2][CH2][SX2H1]")


GENERAL_FORMULA_ENTRIES: dict[str, tuple[str, int, callable]] = {
    # entry id           description                                sched  fn
    "1A.1":  ("O-Alkyl alkylphosphonofluoridate (G-series)",           1, m_1A1),
    "1A.2":  ("O-Alkyl N,N-dialkylphosphoramidocyanidate (Tabun)",     1, m_1A2),
    "1A.3":  ("O-Alkyl S-dialkylaminoethyl alkylphosphonothiolate (V)",1, m_1A3),
    "1A.13": ("P-alkyl phosphonamidic fluoride (Novichok)",            1, m_1A13),
    "1A.14": ("O-alkyl phosphoramidofluoridate (Novichok)",            1, m_1A14),
    "1A.15": ("Bis(dialkylamino)methylene phosphonamidofluoridate",    1, m_1A15),
    "1B.9":  ("Alkylphosphonyl difluoride",                            1, m_1B9),
    "1B.10": ("O-Alkyl O-dialkylaminoethyl alkylphosphonite (QL)",     1, m_1B10),
    "2B.4":  ("Single Me/Et/Pr group bonded to phosphorus",            2, m_2B4),
    "2B.5":  ("N,N-Dialkylphosphoramidic dihalide",                    2, m_2B5),
    "2B.6":  ("Dialkyl N,N-dialkylphosphoramidate",                    2, m_2B6),
    "2B.10": ("N,N-Dialkylaminoethyl-2-chloride",                      2, m_2B10),
    "2B.11": ("N,N-Dialkylaminoethane-2-ol",                           2, m_2B11),
    "2B.12": ("N,N-Dialkylaminoethane-2-thiol",                        2, m_2B12),
}


# ══════════════════════════════════════════════════════════════════════════
#  discretely listed chemicals
# ══════════════════════════════════════════════════════════════════════════
DISCRETE_CHEMICALS: dict[str, tuple[str, int, str]] = {
    # SMILES                              name                        sched  entry
    "ClCSCCCl":            ("2-Chloroethyl chloromethyl sulfide",       1, "1A.4"),
    "ClCCSCCCl":           ("Mustard gas (HD)",                         1, "1A.4"),
    "ClCCSCSCCCl":         ("Bis(2-chloroethylthio)methane",            1, "1A.4"),
    "ClCCSCCSCCCl":        ("Sesquimustard (Q)",                        1, "1A.4"),
    "ClCCSCCCSCCCl":       ("1,3-Bis(2-chloroethylthio)-n-propane",     1, "1A.4"),
    "ClCCSCCCCSCCCl":      ("1,4-Bis(2-chloroethylthio)-n-butane",      1, "1A.4"),
    "ClCCSCCCCCSCCCl":     ("1,5-Bis(2-chloroethylthio)-n-pentane",     1, "1A.4"),
    "ClCCSCOCSCCCl":       ("Bis(2-chloroethylthiomethyl)ether",        1, "1A.4"),
    "ClCCSCCOCCSCCCl":     ("O-Mustard (T)",                            1, "1A.4"),
    "ClC=C[As](Cl)Cl":     ("Lewisite 1",                               1, "1A.5"),
    "ClC=C[As](C=CCl)Cl":  ("Lewisite 2",                               1, "1A.5"),
    "ClC=C[As](C=CCl)C=CCl": ("Lewisite 3",                             1, "1A.5"),
    "CCN(CCCl)CCCl":       ("Nitrogen mustard HN-1",                    1, "1A.6"),
    "CN(CCCl)CCCl":        ("Nitrogen mustard HN-2",                    1, "1A.6"),
    "ClCCN(CCCl)CCCl":     ("Nitrogen mustard HN-3",                    1, "1A.6"),
    "CC(C)OP(=O)(C)Cl":    ("Chlorosarin",                              1, "1B.11"),
    "CC(OP(=O)(C)Cl)C(C)(C)C": ("Chlorosoman",                          1, "1B.12"),
    "CCOP(=O)(OCC)SCCN(CC)CC": ("Amiton (VG)",                          2, "2A.1"),
    "FC(F)=C(C(F)(F)F)C(F)(F)F": ("PFIB",                               2, "2A.2"),
    "OC(C(=O)OC1CN2CCC1CC2)(c1ccccc1)c1ccccc1": ("BZ",                  2, "2A.3"),
    "Cl[As](Cl)Cl":        ("Arsenic trichloride",                      2, "2B.7"),
    "OC(C(=O)O)(c1ccccc1)c1ccccc1": ("Benzilic acid",                   2, "2B.8"),
    "OC1CN2CCC1CC2":       ("Quinuclidin-3-ol",                         2, "2B.9"),
    "OCCSCCO":             ("Thiodiglycol",                             2, "2B.13"),
    "CC(O)C(C)(C)C":       ("Pinacolyl alcohol",                        2, "2B.14"),
    "ClC(Cl)=O":           ("Phosgene",                                 3, "3A.1"),
    "ClC#N":               ("Cyanogen chloride",                        3, "3A.2"),
    "C#N":                 ("Hydrogen cyanide",                         3, "3A.3"),
    "[O-][N+](=O)C(Cl)(Cl)Cl": ("Chloropicrin",                         3, "3A.4"),
    "ClP(Cl)(Cl)=O":       ("Phosphorus oxychloride",                   3, "3B.5"),
    "ClP(Cl)Cl":           ("Phosphorus trichloride",                   3, "3B.6"),
    "ClP(Cl)(Cl)(Cl)Cl":   ("Phosphorus pentachloride",                 3, "3B.7"),
    "COP(OC)OC":           ("Trimethyl phosphite",                      3, "3B.8"),
    "CCOP(OCC)OCC":        ("Triethyl phosphite",                       3, "3B.9"),
    "COP(=O)OC":           ("Dimethyl phosphite",                       3, "3B.10"),
    "CCOP(=O)OCC":         ("Diethyl phosphite",                        3, "3B.11"),
    "ClSSCl":              ("Sulfur monochloride",                      3, "3B.12"),
    "ClSCl":               ("Sulfur dichloride",                        3, "3B.13"),
    "ClS(Cl)=O":           ("Thionyl chloride",                         3, "3B.14"),
    "CCN(CCO)CCO":         ("Ethyldiethanolamine",                      3, "3B.15"),
    "CN(CCO)CCO":          ("Methyldiethanolamine",                     3, "3B.16"),
    "OCCN(CCO)CCO":        ("Triethanolamine",                          3, "3B.17"),
}

# Annex exemptions, keyed by InChIKey connectivity block
EXEMPTIONS: dict[str, tuple[str, str]] = {
    "CCOP(=S)(CC)Sc1ccccc1": ("Fonofos", "2B.4"),
    "CN(C)CCO":              ("N,N-Dimethylaminoethanol", "2B.11"),
    "CCN(CC)CCO":            ("N,N-Diethylaminoethanol", "2B.11"),
}


def _skeleton(smiles_or_mol) -> str | None:
    """Connectivity block of the standard InChIKey (stereo/salt-insensitive)."""
    mol = (Chem.MolFromSmiles(smiles_or_mol)
           if isinstance(smiles_or_mol, str) else smiles_or_mol)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key.split("-")[0] if key else None


_DISCRETE_BY_SKEL: dict[str, tuple[str, int, str]] = {}
_EXEMPT_BY_SKEL: dict[str, tuple[str, str]] = {}
_LOAD_ERRORS: list[str] = []

for _smi, _meta in DISCRETE_CHEMICALS.items():
    _k = _skeleton(_smi)
    if _k is None:
        _LOAD_ERRORS.append(f"discrete SMILES failed to parse: {_smi}")
    else:
        _DISCRETE_BY_SKEL[_k] = _meta

for _smi, _meta in EXEMPTIONS.items():
    _k = _skeleton(_smi)
    if _k is None:
        _LOAD_ERRORS.append(f"exemption SMILES failed to parse: {_smi}")
    else:
        _EXEMPT_BY_SKEL[_k] = _meta


# ══════════════════════════════════════════════════════════════════════════
#  auxiliary CWC-derived scaffold motifs
# ══════════════════════════════════════════════════════════════════════════
#  Broader than the statutory formulas: these encode the parent scaffolds the
#  Schedules are built around, and serve as the substructure-fingerprint block
#  of the descriptor set.
SCAFFOLD_MOTIFS: dict[str, str] = {
    "sca_P_fluoridate":    "[#15X4](=[OX1])F",
    "sca_P_alkyl_direct":  "[#15]~[CX4]",
    "sca_P_aryl_direct":   "[#15]~c",
    "sca_P_cyanide":       "[#15][CX2]#[NX1]",
    "sca_P_amidine":       "[#15][NX2]=[CX3][NX3]",
    "sca_P_thiolate_amine":"[#15](=[OX1])[SX2][CH2][CH2][NX3]",
    "sca_P_thiono":        "[#15]=[SX1]",
    "sca_P_dithio":        "[#15](=[SX1])[SX2]",
    "sca_P_halide":        "[#15X4](=[OX1])[F,Cl,Br,I]",
    "sca_P_amide":         "[#15X4](=[OX1])[NX3]",
    "sca_P_aryl_ester":    "[#15](=[O,S])[OX2]c",
    "sca_P_III":           "[#15X3]([OX2])([OX2])[OX2]",
    "sca_chloroethyl_S":   "[Cl][CH2][CH2][SX2]",
    "sca_chloroethyl_N":   "[Cl][CH2][CH2][NX3]",
    "sca_arsenical":       "[As]",
    "sca_vinyl_arsenical": "[As][CX3]=[CX3][Cl]",
    "sca_aminoethyl":      "[NX3]([CX4])([CX4])[CH2][CH2][F,Cl,Br,I,OX2H1,SX2H1]",
    # nitro groups are written charge-separated in canonical SMILES, so both
    # the neutral pentavalent and the zwitterionic form must be accepted
    "sca_nitroaryl_ester": ("[#15](=[O,S])[OX2]c1ccc(cc1)"
                            "[$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[OX1-])]"),
    # non-phosphorus parent scaffolds of the Schedule 1 vesicant entries and
    # of the corresponding non-phosphorus surrogate compounds
    "sca_thioether":       "[#16X2]([#6])[#6]",
    "sca_disulfide":       "[#16X2][#16X2]",
    "sca_thiophene_ring":  "[#16X2r5]",
    "sca_beta_haloalkyl":  "[F,Cl,Br,I][CX4][CX4]",
    "sca_aryl_ester":      "[cX3][CX3](=[OX1])[OX2][#6]",
    "sca_nitro_aromatic":  ("[c][$([NX3](=[OX1])=[OX1]),"
                            "$([NX3+](=[OX1])[OX1-])]"),
    "sca_acyl_halide":     "[CX3](=[OX1])[F,Cl,Br,I]",
    "sca_nitrile":         "[CX2]#[NX1]",
}
_MOTIF_QUERIES = {k: Chem.MolFromSmarts(v) for k, v in SCAFFOLD_MOTIFS.items()}
for _k, _q in _MOTIF_QUERIES.items():
    if _q is None:
        _LOAD_ERRORS.append(f"motif SMARTS failed to parse: {_k}")

MOTIF_NAMES = list(SCAFFOLD_MOTIFS)


def load_errors() -> list[str]:
    """Any pattern that failed to compile at import time (should be empty)."""
    return list(_LOAD_ERRORS)


# ══════════════════════════════════════════════════════════════════════════
#  public API
# ══════════════════════════════════════════════════════════════════════════
def classify(smiles: str) -> dict:
    """Assign CWC Schedule status to one structure.

    Returns a dict with:
      parsed            bool  - RDKit accepted the SMILES
      entries           list  - every matching Annex entry id
      schedule          int   - 0 (unlisted) or the most restrictive of 1/2/3
      exempt            str   - Annex exemption applied, if any
      discrete_name     str   - name of the discretely listed chemical matched
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"parsed": False, "entries": [], "schedule": 0,
                "exempt": "", "discrete_name": ""}

    skel = _skeleton(mol)
    entries: list[str] = []
    schedules: list[int] = []
    discrete_name = ""

    if skel and skel in _DISCRETE_BY_SKEL:
        name, sched, entry = _DISCRETE_BY_SKEL[skel]
        entries.append(entry)
        schedules.append(sched)
        discrete_name = name

    general_hits: list[tuple[str, int]] = []
    for entry, (_desc, sched, fn) in GENERAL_FORMULA_ENTRIES.items():
        try:
            if fn(mol):
                general_hits.append((entry, sched))
        except Exception:                                  # pragma: no cover
            continue

    in_schedule_1 = any(s == 1 for s in schedules) or any(
        s == 1 for _e, s in general_hits)

    exempt = ""
    if skel and skel in _EXEMPT_BY_SKEL:
        exempt_name, exempt_entry = _EXEMPT_BY_SKEL[skel]
        exempt = f"{exempt_name} ({exempt_entry})"
    else:
        exempt_entry = None

    for entry, sched in general_hits:
        # Schedule 1 takes precedence over the 2B.4 general formula
        if entry == "2B.4" and in_schedule_1:
            continue
        # honour Annex exemptions for the specific entry they apply to
        if exempt and exempt_entry == entry:
            continue
        entries.append(entry)
        schedules.append(sched)

    entries = sorted(dict.fromkeys(entries))
    schedule = min(schedules) if schedules else 0
    return {"parsed": True, "entries": entries, "schedule": schedule,
            "exempt": exempt, "discrete_name": discrete_name}


def motif_flags(smiles: str) -> dict[str, int]:
    """Binary CWC-derived scaffold motif fingerprint for one structure."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {k: 0 for k in MOTIF_NAMES}
    return {k: int(bool(mol.GetSubstructMatches(_MOTIF_QUERIES[k])))
            for k in MOTIF_NAMES}


def schedule_features(smiles: str) -> dict[str, int]:
    """Full CWC substructure-fingerprint block for the descriptor set.

    Per-entry indicators for the general-formula families, three
    Schedule-level indicators, and the scaffold motif block.
    """
    res = classify(smiles)
    feats: dict[str, int] = {}
    for entry in GENERAL_FORMULA_ENTRIES:
        feats["cwc_" + entry.replace(".", "_")] = int(entry in res["entries"])
    feats["cwc_sch1"] = int(res["schedule"] == 1)
    feats["cwc_sch2"] = int(res["schedule"] == 2)
    feats["cwc_sch3"] = int(res["schedule"] == 3)
    feats["cwc_listed"] = int(res["schedule"] > 0)
    feats.update(motif_flags(smiles))
    return feats


CWC_FEATURE_NAMES = (
    ["cwc_" + e.replace(".", "_") for e in GENERAL_FORMULA_ENTRIES]
    + ["cwc_sch1", "cwc_sch2", "cwc_sch3", "cwc_listed"]
    + MOTIF_NAMES
)
