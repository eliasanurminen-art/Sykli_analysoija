"""
Kandi 4.0 – Sykli-Analysoija (SA)
=================================

Laskee syklisen kaukolämpöverkon tilavuusvirrat, painehäviöt ja
painekorkeudet. Verkko luetaan Puun_istutus-ohjelman (PI) tallentamasta
tiedostosta network_data.npz.

Laskennan vaiheet (kandityön luvut suluissa):
  1. Cycle_Finder           – leveyshaku: solmujen ja linkkien tasot,
                              puumallin linkit ja SML:t           (2.3, 2.3.1)
  2. Etsi_syklin_linkit     – sykliin kuuluvat linkit ja
                              syklimatriisi C (l x k)              (2.3.1)
  3. Alkuarvaus             – tilavuusvirrat lehdistä kohti 0-solmua
                              massansäilymislailla                  (2.6)
  4. Newton_Iteration       – Newton-Raphson-askel syklien
                              residuaalista F = C^T h               (2.4)
  5. Vaimennin              – askeleen puolitus (1/2)^x, jos F
                              ei pienene                            (2.5)
  6. Laske_painekorkeudet   – H solmuihin konvergenssin jälkeen     (2.1.3)

Merkinnät:
  l = linkkien määrä, m = solmujen määrä, k = syklien määrä
  Q  (l,)   tilavuusvirrat [m^3/s], positiivinen = nimellissuunta u -> v
  h  (l,)   painehäviöt [m],  h = (K_l + K_m) * Q * |Q|
  h' (l,)   derivaatta dh/dQ = 2 * (K_l + K_m) * |Q|
  C  (l,k)  syklimatriisi, alkio +1 / -1 linkin suunnan mukaan
  F  (k,)   painehäviöiden summa syklin ympäri, F = C^T h  (tavoite 0)

Q_features-matriisin sarakkeet (luku 3.3):
  0 L, 1 D, 2 A, 3 karheus, 4 u, 5 v, 6 Q, 7 dE, 8 h_loss
"""

from __future__ import annotations

from collections import deque

import numpy as np
from CoolProp.CoolProp import PropsSI
from fluids.friction import friction_factor

# ----------------------------------------------------------------------------
# PARAMETRIT
# ----------------------------------------------------------------------------
TIEDOSTO = "network_data.npz"
JUURI = 0                     # 0-solmu (syöttö)
Q_KULUTUS = 0.0006            # m^3/s, yhden kuluttajan kulutus (luku 2.6)
G = 9.81                      # m/s^2
T_VESI = 80.0 + 273.15        # K, vakiolämpötila
P_JUURI = 1.6e6               # Pa, 0-solmun paine (painekorkeuksien laskentaan)
XI_PAIKALLIS = 1.5            # paikallishäviökerroin (tasoristeys, luku 2.1)
TOL = 1e-3                    # m, max |F_k| < TOL kaikissa sykleissä
MAX_ITER = 50
MAX_PUOLITUKSET = 10          # lambda > 10 -> iteraatio ei onnistu

# Sarakeindeksit Q_features / edge_features -matriisissa
C_L, C_D, C_A, C_EPS, C_U, C_V, C_Q, C_DE, C_H = range(9)


# ============================================================================
# 0) APUFUNKTIOT
# ============================================================================
def otsikko(teksti: str):
    print("\n" + "=" * 70)
    print(teksti)
    print("=" * 70)


def veden_ominaisuudet(T: float = T_VESI, P: float = P_JUURI):
    """Tiheys ja kinemaattinen viskositeetti vakiolämpötilassa (luku 2.2)."""
    rho = PropsSI("D", "T", T, "P", P, "Water")
    mu = PropsSI("V", "T", T, "P", P, "Water")
    return rho, mu / rho


def lue_verkko(tiedosto: str = TIEDOSTO):
    data = np.load(tiedosto)
    edges = data["edges"].astype(np.int64)
    edge_features = data["edge_features"][:, :6].astype(np.float64)
    n_solmut = int(edges.max()) + 1

    if "kuluttajat" in data.files:
        kuluttajat = data["kuluttajat"].astype(np.int64)
    else:
        # Vanha tiedostomuoto: kuluttajat = solmut, joista ei lähde linkkejä
        lahtee = np.zeros(n_solmut, dtype=bool)
        lahtee[edges[:, 0]] = True
        kuluttajat = np.where(~lahtee)[0]
        print("[huom] network_data.npz ei sisällä kuluttajia, käytetään nimellissuunnan lehtiä.")
    return edges, edge_features, n_solmut, kuluttajat


# ============================================================================
# 1) CYCLE_FINDER – leveyshaku (luvut 2.3 ja 2.3.1)
# ============================================================================
def Cycle_Finder(edges: np.ndarray, n_solmut: int, juuri: int = JUURI) -> dict:
    """
    Leveyshaku 0-solmusta linkkien nimellissuuntaa pitkin.

    * Kun linkki johtaa uuteen solmuun, linkki kuuluu puumalliin ja
      loppusolmu saa tason = alkusolmun taso + 1.
    * Kun linkki johtaa jo käsiteltyyn solmuun, linkki on syklin
      merkkaava linkki (SML).
    * Samalla lasketaan, montako linkkiä tuo vettä solmuun (n_sisaan) ja
      montako vie (n_ulos). Näitä tarvitaan alkuarvauksessa.

    Linkin taso = sen loppusolmun taso (0-solmusta lähtevät linkit ovat
    tasolla 1). SML:ille taso merkitään -1, koska ne eivät kuulu puuhun.
    """
    l = len(edges)
    ulos = [[] for _ in range(n_solmut)]
    for i, (u, v) in enumerate(edges):
        ulos[u].append(i)

    solmu_taso = np.full(n_solmut, -1, dtype=int)
    linkki_taso = np.full(l, -1, dtype=int)
    vanhempi_linkki = np.full(n_solmut, -1, dtype=int)   # puun linkki, joka tuo solmuun
    kasitelty = np.zeros(n_solmut, dtype=bool)

    solmu_taso[juuri] = 0
    kasitelty[juuri] = True
    jono = deque([juuri])
    puun_linkit, sml = [], []

    while jono:
        u = jono.popleft()
        for i in ulos[u]:
            v = edges[i, 1]
            if not kasitelty[v]:
                kasitelty[v] = True
                solmu_taso[v] = solmu_taso[u] + 1
                linkki_taso[i] = solmu_taso[v]
                vanhempi_linkki[v] = i
                puun_linkit.append(i)
                jono.append(v)
            else:
                sml.append(i)

    if not kasitelty.all():
        puuttuvat = np.where(~kasitelty)[0]
        raise RuntimeError(f"Solmuihin {puuttuvat} ei päästä 0-solmusta nimellissuuntaa pitkin.")

    n_sisaan = np.bincount(edges[:, 1], minlength=n_solmut)
    n_ulos = np.bincount(edges[:, 0], minlength=n_solmut)

    # solmu_data: sarake 0 = taso, 1 = n_sisään, 2 = n_ulos
    solmu_data = np.column_stack((solmu_taso, n_sisaan, n_ulos))

    tasot = [np.where(solmu_taso == t)[0] for t in range(solmu_taso.max() + 1)]

    return {
        "tasot": tasot,
        "solmu_data": solmu_data,
        "linkki_taso": linkki_taso,
        "vanhempi_linkki": vanhempi_linkki,
        "puun_linkit": np.array(puun_linkit, dtype=int),
        "sml": np.array(sml, dtype=int),
    }


# ============================================================================
# 2) SYKLIEN LINKIT JA SYKLIMATRIISI C (luku 2.3.1)
# ============================================================================
def Etsi_syklin_linkit(edges: np.ndarray, verkko: dict) -> dict:
    """
    Jokaiselle SML:lle (alku a, loppu b) etsitään sykli puumallista:

      reitti[0] alkaa SML:n matalammalla tasolla olevasta päätesolmusta,
      reitti[1] korkeammalla tasolla olevasta.
      1) reitti[1]:een lisätään ensin (tasoero) kpl linkkejä ylöspäin,
         jotta molemmat reitit ovat samalla tasolla.
      2) Sen jälkeen noustaan molempia reittejä yhtä aikaa: seuraava linkki
         on aina se puun linkki, joka päättyy edellisen linkin alkusolmuun.
      3) Lopetetaan, kun reitit päätyvät samaan solmuun.

    Syklin nimellissuunta = SML:n suunta a -> b. Sykli kulkee siis
        a -> b  (SML, +1)
        b -> ... -> yhteinen solmu   (puun linkit vastavirtaan, -1)
        yhteinen solmu -> ... -> a   (puun linkit myötävirtaan, +1)
    eli a:sta alkavan reitin linkit saavat +1 ja b:stä alkavan -1.
    """
    l = len(edges)
    solmu_taso = verkko["solmu_data"][:, 0]
    vanhempi_linkki = verkko["vanhempi_linkki"]
    sml_lista = verkko["sml"]
    k = len(sml_lista)

    C = np.zeros((l, k), dtype=float)
    syklit = []

    for j, s in enumerate(sml_lista):
        a, b = edges[s]
        # 0 = matalampi taso, 1 = korkeampi taso
        paat = sorted([a, b], key=lambda x: solmu_taso[x])
        reitti = [[], []]
        solmu = list(paat)

        # 1) Tasataan tasot
        tasoero = solmu_taso[paat[1]] - solmu_taso[paat[0]]
        for _ in range(tasoero):
            e = vanhempi_linkki[solmu[1]]
            reitti[1].append(e)
            solmu[1] = edges[e, 0]

        # 2) Noustaan molempia reittejä, kunnes yhteinen solmu löytyy
        while solmu[0] != solmu[1]:
            for r in (0, 1):
                e = vanhempi_linkki[solmu[r]]
                reitti[r].append(e)
                solmu[r] = edges[e, 0]
        yhteinen = solmu[0]

        # 3) Etumerkit syklin nimellissuunnan mukaan
        etumerkki = {a: +1.0, b: -1.0}
        C[s, j] = +1.0
        for r in (0, 1):
            for e in reitti[r]:
                C[e, j] = etumerkki[paat[r]]

        # Linkit kulkujärjestyksessä: SML, b:n reitti, a:n reitti käännettynä
        reitti_a = reitti[0] if paat[0] == a else reitti[1]
        reitti_b = reitti[1] if paat[0] == a else reitti[0]
        jarjestys = [s] + reitti_b + reitti_a[::-1]
        syklit.append(np.array(jarjestys, dtype=int))

        print(f"  Sykli {j}: SML {s} = ({a}->{b}), yhteinen solmu {yhteinen}, "
              f"{len(jarjestys)} linkkiä")
        print(f"    linkit (etumerkki): "
              + ", ".join(f"{e}({'+' if C[e, j] > 0 else '-'})" for e in jarjestys))

    return {
        "num_cycles": k,
        "cycles": syklit,
        "cycle_lengths": np.array([len(c) for c in syklit], dtype=int),
        "defining_edges": edges[sml_lista] if k else np.empty((0, 2), dtype=int),
        "C": C,
    }


# ============================================================================
# 3) ALKUARVAUS (luku 2.6)
# ============================================================================
def topologinen_jarjestys(edges: np.ndarray, n_solmut: int) -> np.ndarray:
    """Kahnin algoritmi: jokainen linkki u -> v tulee ennen v:tä.
    (Pelkkä käänteinen leveyshaku ei riitä, koska SML voi viedä vettä
    syvemmältä tasolta matalammalle.)"""
    sisaan = np.bincount(edges[:, 1], minlength=n_solmut)
    ulos = [[] for _ in range(n_solmut)]
    for u, v in edges:
        ulos[u].append(v)
    jono = deque(np.where(sisaan == 0)[0])
    tulos = []
    while jono:
        s = jono.popleft()
        tulos.append(s)
        for v in ulos[s]:
            sisaan[v] -= 1
            if sisaan[v] == 0:
                jono.append(v)
    if len(tulos) != n_solmut:
        raise RuntimeError("Nimellissuunnissa on kehä – alkuarvausta ei voi muodostaa.")
    return np.array(tulos)


def Alkuarvaus(edges: np.ndarray, n_solmut: int, kuluttajat: np.ndarray,
               q_kulutus: float = Q_KULUTUS) -> tuple[np.ndarray, np.ndarray]:
    """
    Massansäilymislaki solmuittain:  Q_sisään - Q_ulos = q_solmu.

    Edetään lehdistä kohti 0-solmua (käänteinen topologinen järjestys):
      virta_solmussa = kulutus + sum(Q lähtevät linkit)
      Q jokaiseen saapuvaan linkkiin = virta_solmussa / n_sisään
    0-solmun tuotanto saadaan lopuksi samasta taseesta.
    """
    q = np.zeros(n_solmut)
    q[kuluttajat] = q_kulutus

    l = len(edges)
    Q = np.full(l, np.nan)
    sisaan_linkit = [[] for _ in range(n_solmut)]
    ulos_linkit = [[] for _ in range(n_solmut)]
    for i, (u, v) in enumerate(edges):
        ulos_linkit[u].append(i)
        sisaan_linkit[v].append(i)

    for solmu in topologinen_jarjestys(edges, n_solmut)[::-1]:
        virta = q[solmu] + sum(Q[i] for i in ulos_linkit[solmu])
        if sisaan_linkit[solmu]:
            for i in sisaan_linkit[solmu]:
                Q[i] = virta / len(sisaan_linkit[solmu])
        else:
            q[solmu] = -virta          # 0-solmu: tuotanto (negatiivinen kulutus)

    return Q, q


def massatase(edges: np.ndarray, Q: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Palauttaa solmukohtaisen virheen Q_sisään - Q_ulos - q (pitäisi olla 0)."""
    virhe = -q.copy()
    np.add.at(virhe, edges[:, 1], Q)
    np.add.at(virhe, edges[:, 0], -Q)
    return virhe


# ============================================================================
# 4) PAINEHÄVIÖT (luvut 2.1 ja 2.2)
# ============================================================================
_kitka = np.vectorize(friction_factor)


def Painehavio(Q: np.ndarray, edge_features: np.ndarray, nu: float):
    """
    h  = (K_l + K_m) * Q * |Q|      (etumerkki seuraa virtaussuuntaa)
    h' = 2 * (K_l + K_m) * |Q|
    K_l = f * L / (2 g A^2 D)       (Darcy–Weisbach, f Colebrook–White)
    K_m = xi / (2 g A^2)            (paikallishäviö)
    """
    L = edge_features[:, C_L]
    D = edge_features[:, C_D]
    A = edge_features[:, C_A]
    eps = edge_features[:, C_EPS]

    v = Q / A
    Re = np.maximum(np.abs(v) * D / nu, 1.0)    # vältetään Re = 0
    f = _kitka(Re=Re, eD=eps / D)

    K_l = f * L / (2 * G * A ** 2 * D)
    K_m = XI_PAIKALLIS / (2 * G * A ** 2)
    K = K_l + K_m

    h = K * Q * np.abs(Q)
    dh = 2 * K * np.abs(Q)
    return h, dh, {"v": v, "Re": Re, "f": f}


def Syklien_residuaali(C: np.ndarray, h: np.ndarray) -> np.ndarray:
    """F_k = sum_l h_l * C_lk  (kaava 2.4.5)."""
    return C.T @ h


# ============================================================================
# 5) NEWTON-ITERAATIO JA VAIMENNIN (luvut 2.4 ja 2.5)
# ============================================================================
def Newton_Iteration(C: np.ndarray, h: np.ndarray, dh: np.ndarray):
    """
    Yksi Newton-Raphson-askel syklivirroille.

    Kun syklin k ympäri lisätään korjaus dQ_k, linkin virta muuttuu
    Q_l -> Q_l - C_lk dQ_k. Linearisoimalla F(Q) saadaan yhtälöryhmä
        (C^T diag(h') C) dQ_sykli = F
    Yhdellä syklillä tämä on täsmälleen kaava 2.4.11:
        dQ = F / sum(h' |C|).
    Palauttaa syklikorjaukset dQ_sykli (k,) ja linkkikorjaukset C dQ_sykli (l,).
    """
    F = Syklien_residuaali(C, h)
    J = C.T @ (dh[:, None] * C)               # (k x k)
    dQ_sykli = np.linalg.solve(J, F)
    dQ_linkki = C @ dQ_sykli
    return dQ_sykli, dQ_linkki


def Vaimennin(Q: np.ndarray, dQ_linkki: np.ndarray, F_vanha: np.ndarray,
              C: np.ndarray, edge_features: np.ndarray, nu: float):
    """
    Kokeillaan askelta Q_uusi = Q - (1/2)^x * dQ, x = 0, 1, 2, ...
    Hyväksytään ensimmäinen, jolla max|F_uusi| < max|F_vanha| (kaava 2.5.1).
    Jos x > MAX_PUOLITUKSET, iteraatio ei onnistu.
    """
    norm_vanha = np.max(np.abs(F_vanha))
    for x in range(MAX_PUOLITUKSET + 1):
        kerroin = 0.5 ** x
        Q_uusi = Q - kerroin * dQ_linkki
        h, dh, _ = Painehavio(Q_uusi, edge_features, nu)
        F_uusi = Syklien_residuaali(C, h)
        if np.max(np.abs(F_uusi)) < norm_vanha:
            return Q_uusi, h, dh, F_uusi, x
        print(f"      vaimennus: x={x}, max|F|={np.max(np.abs(F_uusi)):.3e} "
              f">= {norm_vanha:.3e} -> puolitetaan askel")
    raise RuntimeError(f"Iteraatio ei suppene: askelta puolitettu yli {MAX_PUOLITUKSET} kertaa.")


# ============================================================================
# 6) PAINEKORKEUDET (kaava 2.1.3)
# ============================================================================
def Laske_painekorkeudet(edges: np.ndarray, h: np.ndarray, verkko: dict,
                         n_solmut: int, H0: float) -> np.ndarray:
    """H_v = H_u - h_l puun linkkejä pitkin 0-solmusta tasoittain."""
    H = np.full(n_solmut, np.nan)
    H[JUURI] = H0
    for taso in verkko["tasot"][1:]:
        for v in taso:
            i = verkko["vanhempi_linkki"][v]
            H[v] = H[edges[i, 0]] - h[i]
    return H


# ============================================================================
# PÄÄOHJELMA
# ============================================================================
def main():
    rho, nu = veden_ominaisuudet()

    # ---------------------------------------------------------------- verkko
    otsikko("1) VERKON LUKEMINEN")
    edges, edge_features, n_solmut, kuluttajat = lue_verkko()
    l = len(edges)
    print(f"Solmuja m = {n_solmut}, linkkejä l = {l}, kuluttajia = {len(kuluttajat)}")
    print(f"Vesi {T_VESI - 273.15:.0f} °C: rho = {rho:.1f} kg/m3, nu = {nu:.3e} m2/s")

    # ---------------------------------------------------------- Cycle_Finder
    otsikko("2) CYCLE_FINDER – tasot ja SML:t")
    verkko = Cycle_Finder(edges, n_solmut)
    print(f"Tasoja: {len(verkko['tasot'])}")
    for t, solmut in enumerate(verkko["tasot"]):
        print(f"  taso {t:2d}: {len(solmut):3d} solmua  {solmut.tolist()}")
    print(f"Puumallin linkit: {len(verkko['puun_linkit'])} (pitäisi olla m-1 = {n_solmut - 1})")
    print(f"SML:t (indeksit): {verkko['sml'].tolist()}  -> syklejä {len(verkko['sml'])}")
    assert len(verkko["puun_linkit"]) == n_solmut - 1, "Puumallin linkkimäärä ei täsmää!"

    # ---------------------------------------------------------------- syklit
    otsikko("3) SYKLIEN LINKIT JA SYKLIMATRIISI C")
    cycle_db = Etsi_syklin_linkit(edges, verkko)
    C = cycle_db["C"]
    print(f"C-matriisin koko: {C.shape} (linkit x syklit)")

    # ------------------------------------------------------------ alkuarvaus
    otsikko("4) ALKUARVAUS (lehdistä kohti 0-solmua)")
    Q, q = Alkuarvaus(edges, n_solmut, kuluttajat)
    tase = massatase(edges, Q, q)
    print(f"0-solmun tuotanto = {-q[JUURI]:.5f} m3/s "
          f"(= {len(kuluttajat)} kuluttajaa x {Q_KULUTUS} m3/s = {len(kuluttajat) * Q_KULUTUS:.5f})")
    print(f"Massataseen suurin virhe solmuissa: {np.max(np.abs(tase)):.2e} m3/s")
    h, dh, apu = Painehavio(Q, edge_features, nu)
    F = Syklien_residuaali(C, h)
    print(f"Virtausnopeudet: {np.abs(apu['v']).min():.3f}–{np.abs(apu['v']).max():.3f} m/s, "
          f"Re: {apu['Re'].min():.0f}–{apu['Re'].max():.0f}")
    for j, s in enumerate(verkko["sml"]):
        print(f"  Sykli {j}: SML {s} Q0 = {Q[s]:.6f} m3/s, F0 = {F[j]:.4f} m")

    # ------------------------------------------------------------- iteraatio
    otsikko("5) NEWTON-ITERAATIO (vaimennettu)")
    print(f"{'n':>3} {'max|F| [m]':>12} {'max|dQ_sykli|':>14} {'x':>3}   Q_SML [m3/s]")
    print(f"{0:3d} {np.max(np.abs(F)):12.4e} {'-':>14} {'-':>3}   "
          + ", ".join(f"{Q[s]:.6f}" for s in verkko["sml"]))

    n = 0
    while np.max(np.abs(F)) >= TOL:
        n += 1
        if n > MAX_ITER:
            raise RuntimeError(f"Ei suppenemista {MAX_ITER} iteraatiossa.")
        dQ_sykli, dQ_linkki = Newton_Iteration(C, h, dh)
        Q, h, dh, F, x = Vaimennin(Q, dQ_linkki, F, C, edge_features, nu)
        print(f"{n:3d} {np.max(np.abs(F)):12.4e} {np.max(np.abs(dQ_sykli)):14.4e} {x:3d}   "
              + ", ".join(f"{Q[s]:.6f}" for s in verkko["sml"]))

    print(f"\nSuppeni {n} iteraatiolla: max|F| = {np.max(np.abs(F)):.3e} m < {TOL} m")
    tase = massatase(edges, Q, q)
    print(f"Massatase iteraation jälkeen: suurin virhe {np.max(np.abs(tase)):.2e} m3/s "
          f"(syklikorjaus ei muuta solmujen tasetta)")
    kaantyneet = np.where(Q < 0)[0]
    if kaantyneet.size:
        print(f"Huom: virtaus kääntyi nimellissuuntaa vastaan linkeissä {kaantyneet.tolist()}")

    # -------------------------------------------------------- painekorkeudet
    otsikko("6) PAINEKORKEUDET JA TULOKSET")
    H0 = P_JUURI / (rho * G)
    H = Laske_painekorkeudet(edges, h, verkko, n_solmut, H0)
    dE = h + H[edges[:, 1]] - H[edges[:, 0]]     # linkin energiayhtälön virhe
    print(f"H(0-solmu) = {H0:.2f} m,  pienin H = {np.nanmin(H):.2f} m (solmu {np.nanargmin(H)})")
    print(f"Energiayhtälön virhe dE: puun linkeissä max {np.max(np.abs(dE[verkko['puun_linkit']])):.1e} m, "
          f"SML:issä " + ", ".join(f"{dE[s]:.2e}" for s in verkko["sml"]) + " m (= F)")

    Q_features = np.column_stack((edge_features, Q, dE, h))

    for s in verkko["sml"]:
        u, v = edges[s]
        print(f"SML {s}: H[{u}] = {H[u]:.3f} m, H[{v}] = {H[v]:.3f} m, h = {h[s]:.4f} m, "
              f"Q = {Q[s]:.6f} m3/s")

    nayte = min(30, l - 1)
    print(f"\nLinkki {nayte}: [L, D, A, eps, u, v, Q, dE, h_loss] =")
    print("  ", np.array2string(Q_features[nayte], precision=5, suppress_small=False))

    return Q_features, H, cycle_db


if __name__ == "__main__":
    main()
