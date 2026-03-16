import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 0) Reproduzierbarkeit / Device / Dtype
# ============================================================
# Zweck:
# - Reproduzierbarkeit: Zufallszahlen (NumPy + PyTorch) werden fest „genagelt“,
#   damit Trainingsläufe (weitgehend) vergleichbar sind.
# - Default-Datentyp: float32 (schneller, weniger Speicher; kann aber bei sehr steifen Problemen
#   manchmal weniger stabil sein als float64).
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

# Device-Auswahl:
# - Wenn CUDA verfügbar ist: GPU nutzen (schneller).
# - Sonst: CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Startzeit zur Laufzeitmessung des gesamten Programms
t_total_start = time.perf_counter()

# ============================================================
# 1) Problemparameter
# ============================================================
# Balkenlänge (wird als Skalierungsgröße verwendet)
L = 1.0

# ------------------------------------------------------------
# NEU: Messrauschen-Szenarien in Prozent
# ------------------------------------------------------------
# Messrauschen in Prozent (bezogen auf die Standardabweichung der Messwerte)
# - 0 bedeutet: perfekte (rauschfreie) Messwerte
# - 1,2,5 bedeutet: sigma = (pct/100)*std(w_data_clean)
noise_levels_pct = [0, 1, 2, 5]

def add_noise_percent(y, pct, rng):
    """
    Additives Gaußrauschen mit sigma = (pct/100) * std(y).
    y: Array der Messwerte (Nd,)
    pct: z.B. 1,2,5
    rng: np.random.Generator für Reproduzierbarkeit
    """
    y = np.asarray(y).reshape(-1)
    if pct <= 0:
        return y.copy()
    sigma = (pct / 100.0) * np.std(y)
    return y + sigma * rng.standard_normal(size=y.shape)

# Lastamplituden für die drei Lastfälle (konstant, dreieckförmig, Teilstrecke)
q0_konst = 1.0
q0_dreieck   = 1.0
q0_teilstreckenlast = 1.0

# Parameter für die "Teilstrecke"-Last:
# - Teilstrecke wirkt ungefähr in [a_Teilstrecke, b_Teilstrecke]
# - smooth_w steuert die Glättung der Übergänge (keine harte Sprungstelle -> besser für Autograd)
a_Teilstrecke  = 0.25 * L
b_Teilstrecke  = 0.75 * L
smooth_w = 0.05 * L

# ------------------------------------------------------------
# Skalierung x -> x_hat ∈ [-1,1]
# ------------------------------------------------------------
# Hintergrund:
# Neurale Netzwerke trainieren oft stabiler, wenn Eingaben normiert/skalierend sind.
# Hier wird physikalisches x in [0, L] auf x_hat in [-1, 1] abgebildet.
def x_hat_np(x):
    return (2.0 * x / L) - 1.0

def x_hat_t(x):
    return (2.0 * x / L) - 1.0

# ------------------------------------------------------------
# "True" EI(x) für synthetische Tests
# ------------------------------------------------------------
# EI_true_np(x) definiert die "wahre" Biegesteifigkeit EI(x), mit der Referenzdaten erzeugt werden.
# Das ist der "Ground Truth", den das PINN später rekonstruieren soll.
ei_verlauf = "quadratisch"     #konstant; quadratisch; sinus
def EI_true_np(x):
    alpha = 1.5  # höher -> stärkerer Anstieg zu den Rändern
    return 1.0 + alpha * ((x / L) - 0.5) ** 2      # EI quadratisch
#    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x / L)    # EI Sinus
#    return alpha * np.ones_like(x)                     # EI konstant

# ============================================================
# 2) Lastfunktionen q(x)
# ============================================================
# Für jeden Lastfall gibt es:
# - eine NumPy-Variante (für FD-Referenzlösung)
# - eine Torch-Variante (für PINN-Loss / Autograd)
def q_konst_np(x): return q0_konst * np.ones_like(x)
def q_dreieck_np(x):   return q0_dreieck * (x / L)

# Glättungsfunktion (Sigmoid) für einen "weichen" Teilstrecke:
# Vorteil: differenzierbar, keine harte Diskontinuität
def smooth_step_np(z):
    return 1.0 / (1.0 + np.exp(-z))

# Teilstrecke-Last: ungefähr 1 im Inneren der Teilstrecke, 0 außerhalb, aber mit glatten Übergängen
def q_Teilstrecke_np(x):
    return q0_teilstreckenlast * smooth_step_np((x - a_Teilstrecke) / smooth_w) * smooth_step_np((b_Teilstrecke - x) / smooth_w)

def q_konst_t(x): return q0_konst * torch.ones_like(x)
def q_dreieck_t(x):   return q0_dreieck * (x / L)

def smooth_step_t(z):
    return 1.0 / (1.0 + torch.exp(-z))

def q_Teilstrecke_t(x):
    return q0_teilstreckenlast * smooth_step_t((x - a_Teilstrecke) / smooth_w) * smooth_step_t((b_Teilstrecke - x) / smooth_w)

# Sammeln der Lastfälle in einer Liste:
# Jeder Eintrag: (Name, NumPy-Funktion, Torch-Funktion)
load_cases = [
    ("konstant", q_konst_np, q_konst_t),
    ("dreieck",  q_dreieck_np,   q_dreieck_t),
    ("Teilstrecke",    q_Teilstrecke_np, q_Teilstrecke_t),
]
# Anzahl Lastfälle
K = len(load_cases)


# ============================================================
# 3) Referenzdaten per Finite-Differenzen (synthetische Messdaten)
# ============================================================
# Ziel:
# - Erzeuge synthetische "Messdaten" w(x) für verschiedene Lastfälle,
#   basierend auf der wahren EI_true_np(x).
#
# DGL (Euler-Bernoulli mit variabler Biegesteifigkeit):
#     d²/dx² ( EI(x) * d²w/dx² ) = q(x)
#
# Randbedingungen (einfach gelagert):
# - w(0) = 0, w(L) = 0   (Durchbiegung an den Lagern 0)
# - M(0) = 0, M(L) = 0   (Moment an einfach gelagerten Enden 0)
#   mit M = EI * w''
#
# Hinweis: Das ist ein synthetischer Vorwärtssolver (FD), NICHT Teil des PINNs,
# sondern nur zur Daten- und Benchmark-Erzeugung.

def forward_solve_fd(q_np_func, N=801):
    # FD-Gitter in [0, L]
    x = np.linspace(0.0, L, N)
    h = x[1] - x[0]

    # "wahre" EI-Verteilung und Lastwerte auf dem Gitter
    EI = EI_true_np(x)
    qx = q_np_func(x)

    # Zweite Ableitungsmatrix D2 (klassische zentrale Differenzen im Inneren)
    # D2*w approximiert w''.
    D2 = np.zeros((N, N))
    for i in range(1, N - 1):
        D2[i, i - 1] = 1.0 / h**2
        D2[i, i]     = -2.0 / h**2
        D2[i, i + 1] = 1.0 / h**2

    # Operator entsprechend DGL:
    # (EI*w'')'' ≈ D2 * (EI * (D2*w))
    # Dabei wird EI als Diagonalmatrix umgesetzt (punktweise Multiplikation im Diskreten).
    A = D2 @ (np.diag(EI) @ D2)
    b = qx.copy()

    # BC: w(0)=0, w(L)=0
    # -> Ersetze 1. und letzte Gleichung im LGS durch Dirichlet-Bedingungen
    A[0, :]  = 0.0; A[0, 0]   = 1.0; b[0]  = 0.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0

    # BC: M(0)=0 und M(L)=0  =>  EI * w'' = 0  (one-sided 2nd derivative)
    # Da w(0) und w(L) festgelegt sind, nutzt man für w'' am Rand ein einseitiges Schema.
    # w''(0) ≈ (2w0 - 5w1 + 4w2 - w3)/h^2
    A[1, :] = 0.0
    A[1, 0] =  2.0 * EI[0] / h**2
    A[1, 1] = -5.0 * EI[0] / h**2
    A[1, 2] =  4.0 * EI[0] / h**2
    A[1, 3] = -1.0 * EI[0] / h**2
    b[1] = 0.0

    # w''(L) ≈ (2wN - 5wN-1 + 4wN-2 - wN-3)/h^2
    A[-2, :] = 0.0
    A[-2, -1] =  2.0 * EI[-1] / h**2
    A[-2, -2] = -5.0 * EI[-1] / h**2
    A[-2, -3] =  4.0 * EI[-1] / h**2
    A[-2, -4] = -1.0 * EI[-1] / h**2
    b[-2] = 0.0

    # Lösen des linearen Gleichungssystems A*w = b
    # Ergebnis w: Referenz-Durchbiegung auf dem FD-Gitter
    w = np.linalg.solve(A, b)
    return x, w, EI, qx

def forward_solve_fd_with_EI(EI_on_grid, q_np_func, N=801):
    """
    Wie forward_solve_fd, aber EI wird von außen vorgegeben (Array auf dem FD-Gitter).
    Damit kann man z.B. EI_hat (PINN) in den FD-Vorwärtssolver stecken.
    """
    x = np.linspace(0.0, L, N)
    h = x[1] - x[0]

    EI = np.asarray(EI_on_grid).reshape(-1)
    assert EI.shape[0] == N, "EI_on_grid muss Länge N haben (gleiches FD-Gitter)."

    qx = q_np_func(x)

    D2 = np.zeros((N, N))
    for i in range(1, N - 1):
        D2[i, i - 1] = 1.0 / h**2
        D2[i, i]     = -2.0 / h**2
        D2[i, i + 1] = 1.0 / h**2

    A = D2 @ (np.diag(EI) @ D2)
    b = qx.copy()

    # BC: w(0)=0, w(L)=0
    A[0, :]  = 0.0; A[0, 0]   = 1.0; b[0]  = 0.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0

    # BC: M(0)=0 und M(L)=0  =>  EI*w''=0
    A[1, :] = 0.0
    A[1, 0] =  2.0 * EI[0] / h**2
    A[1, 1] = -5.0 * EI[0] / h**2
    A[1, 2] =  4.0 * EI[0] / h**2
    A[1, 3] = -1.0 * EI[0] / h**2
    b[1] = 0.0

    A[-2, :] = 0.0
    A[-2, -1] =  2.0 * EI[-1] / h**2
    A[-2, -2] = -5.0 * EI[-1] / h**2
    A[-2, -3] =  4.0 * EI[-1] / h**2
    A[-2, -4] = -1.0 * EI[-1] / h**2
    b[-2] = 0.0

    w = np.linalg.solve(A, b)
    return x, w, EI, qx


# Referenzgitter (hier über konstante Last einmal erzeugt, damit x_ref und EI_ref feststehen)
x_ref, _, EI_ref, _ = forward_solve_fd(q_konst_np, N=801)

# Messpunkte (Nd):
# - Aus dem feinen Referenzgitter werden Nd Punkte gleichmäßig ausgewählt.
# - Diese repräsentieren "Messstellen" (Sensing Locations).
Nd = 81
idx_d = np.linspace(0, len(x_ref) - 1, Nd).astype(int)
x_data = x_ref[idx_d]

# Torch-Tensoren: Datenkoordinaten (als x_hat)
# - PINN-Netze bekommen x_hat in [-1,1] als Input
x_d_hat = x_hat_np(x_data).reshape(-1, 1)
x_d_t = torch.tensor(x_d_hat, dtype=torch.float32, device=device)

# Randpunkte (x_hat):
# Für Randbedingungen am Netzinput (da w-Netz mit x_hat arbeitet)
x0_hat = torch.tensor([[-1.0]], dtype=torch.float32, device=device)
xL_hat = torch.tensor([[ 1.0]], dtype=torch.float32, device=device)

# Physikalische Randkoordinaten (für EI am Rand):
# EI-Modell ist in x ∈ [0, L] definiert (physikalischer Raum)
x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
xL = torch.tensor([[L]],   dtype=torch.float32, device=device)

# ============================================================
# 4) Netze für w_k(x_hat)
# ============================================================
# Idee:
# - Für jeden Lastfall k gibt es ein eigenes NN: w_k(x_hat).
# - Das ist sinnvoll, weil w unter unterschiedlichen Lasten unterschiedlich ist,
#   aber EI(x) ist in allen Fällen gleich (Material-/Querschnittseigenschaft).
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        # Aufbau: (Linear -> Aktivierung) x depth, danach Linear-Ausgang
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Vorwärtsdurchlauf: x_hat -> w (Skalar)
        return self.net(x)

# ============================================================
# 5) EI(x) als Control-Points (linear) + Positivität via Softplus
# ============================================================
# EI wird nicht als Netz modelliert, sondern als diskrete Knotenwerte ("Control Points")
# auf einem gleichmäßigen Gitter. Dazwischen: lineare Interpolation.
#
# Vorteile:
# - klare Parametrisierung (m Parameter statt NN-Gewichte)
# - gut interpretierbar
# - weniger Overfitting als sehr flexibles EI-Netz
#
# Positivität:
# EI muss physikalisch > 0 sein. Das wird erzwungen durch:
#   EI = softplus(raw) + eps
# raw ist frei optimierbar; softplus sorgt für Positivität.
class EIControlPoints(nn.Module):
    """
    EI(x) wird über m Knotenwerte EI_i (i=0..m-1) auf gleichmäßigem Grid [0,L]
    definiert und linear interpoliert. Positivität via Softplus.
    """
    def __init__(self, m=9, beta=5.0, init_EI=1.0, eps=1e-6):
        super().__init__()
        self.m = int(m)
        self.beta = beta
        self.softplus = nn.Softplus(beta=beta)
        self.eps = eps

        # Initialisierung:
        # Wir wollen EI ungefähr bei init_EI starten.
        # Da die optimierten Parameter "raw" in Softplus gehen,
        # wird raw_init so gewählt, dass softplus(raw_init)+eps ≈ init_EI.
        target = float(init_EI) - eps
        raw_init = (1.0 / beta) * torch.log(torch.exp(torch.tensor(beta * target, device=device)) - 1.0)
        self.raw = nn.Parameter(raw_init * torch.ones(self.m, device=device))

    def knot_values(self):
        # Gibt die (positiven) EI-Knotenwerte zurück, shape (m,)
        return self.softplus(self.raw) + self.eps

    def forward(self, x):  # x shape (N,1) oder (N,)
        # Input x ist physikalisch in [0, L]
        x1 = x.view(-1)  # (N,)
        # Abbildung auf Knotenskala t ∈ [0, m-1]
        t = (x1 / L) * (self.m - 1)

        # Linker Index für lineare Interpolation
        idx0 = torch.floor(t).to(torch.long)
        # Clamp, damit idx0 immer gültig ist (0 .. m-2)
        idx0 = torch.clamp(idx0, 0, self.m - 2)
        idx1 = idx0 + 1

        # Interpolationsgewicht w ∈ [0,1]
        w = (t - idx0.to(t.dtype))  # in [0,1]
        EI_knots = self.knot_values()

        # Werte an den beiden benachbarten Knoten
        v0 = EI_knots[idx0]
        v1 = EI_knots[idx1]

        # Lineare Interpolation
        EI_x = (1.0 - w) * v0 + w * v1
        return EI_x.view(-1, 1)

# ============================================================
# 6) Autograd helper
# ============================================================
# Zweck:
# Vereinfachter Zugriff auf Ableitungen via torch.autograd.grad.
# create_graph=True:
#   - wichtig, weil wir höhere Ableitungen brauchen (z.B. w'''' indirekt über M_xx)
# retain_graph=True:
#   - verhindert, dass der Graph sofort freigegeben wird; hilfreich, weil wir mehrfach
#     grad() auf Teilgraphen anwenden. (Hat aber Speicherkosten.)
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

# ============================================================
# 7) Collocationpunkte (Strong/Weak)
# ============================================================
# Collocationpunkte x_f dienen der Physik-Loss-Auswertung im Inneren des Gebietes.
# Vorgehen:
# - zufällige Punkte x_f in [0, L]
# - x_f_t: physikalische Koordinate (für EI(x), q(x), und Ableitungen nach x)
# - x_f_hat_t: skalierte Koordinate (für w-Netz-Eingang)
def build_collocation(Nf=2048):
    x_f = np.random.rand(Nf) * L
    x_f_const = torch.tensor(x_f.reshape(-1, 1), dtype=torch.float32, device=device)
    # requires_grad=True ist entscheidend, weil wir Ableitungen nach x brauchen
    x_f_t = x_f_const.clone().detach().requires_grad_(True)
    x_f_hat_t = x_hat_t(x_f_t)
    return x_f_t, x_f_hat_t

import math

def v_sin_and_vxx_sin(x, j):
    """
    v_j(x) = sin(j*pi*x/L)
    v_j''(x) = -(j*pi/L)^2 * sin(j*pi*x/L)
    x: (N,1) physikalisch in [0,L]

    Hintergrund:
    - In der Weak-Form werden Testfunktionen v_j verwendet.
    - Sinusfunktionen passen gut zu einfachen Randbedingungen und bilden eine Basis.
    """
    k = j * math.pi / L
    v = torch.sin(k * x)
    v_xx = -(k**2) * v
    return v, v_xx

# ============================================================
# 8) Loss: Strong + (optional Weak) + Daten + BC + Regularisierung
# ============================================================
# Gesamtkonzept:
# Wir minimieren eine gewichtete Summe verschiedener Terme:
# - Strong Form (PDE-Residual punktweise): (EI*w'')'' - q = 0
# - Weak Form (integrale Residuen gegen Testfunktionen), optional zusätzlich
# - Daten-Loss: w_pred(x_d) passt zu Messwerten
# - Randbedingungen: w=0 und M=EI*w''=0 an x=0,L
# - Regularisierung EI: (dEI/dx)^2 (Glattheit)
# - Regularisierung Knoten ("Modes"): zweite Differenz der EI-Knoten (Krümmungsstrafe)
# - optional w-Glattheit: (w'')^2 (manchmal hilfreich gegen Artefakte)

def loss_all(
    x_f_t, x_f_hat_t,
    lam_phys, lam_data, lam_bc,
    lam_reg_ei, lam_w_smooth,
    lam_modes,
    lam_weak=0.0,
    n_test=8
):
    # Randpunkte frisch (Graph sauber)
    # Hinweis:
    # - wir wollen Ableitungen an den Randpunkten berechnen
    # - daher requires_grad=True auf x0_hat_req / xL_hat_req
    x0_hat_req = x0_hat.clone().detach().requires_grad_(True)
    xL_hat_req = xL_hat.clone().detach().requires_grad_(True)

    # EI am Collocation-Set (physikalischer Raum)
    EI_f = ei_model(x_f_t)

    # EI-Glattheit (dEI/dx)^2
    # - verhindert "Zacken" im rekonstruierten EI
    if lam_reg_ei > 0.0:
        EI_x = grad(EI_f, x_f_t)
        Lreg_ei = torch.mean(EI_x**2)
    else:
        Lreg_ei = torch.tensor(0.0, device=device)

    # "Modes"-Regularisierung als Krümmungsstrafe auf Knoten (2. Differenz)
    # - diskrete zweite Ableitung der Knotensequenz
    # - drückt hochfrequente Oszillationen in den Knotenwerten weg
    if lam_modes > 0.0 and ei_model.m >= 3:
        EI_k = ei_model.knot_values()  # (m,)
        d2 = EI_k[2:] - 2.0 * EI_k[1:-1] + EI_k[:-2]
        Lreg_modes = torch.mean(d2**2)
    else:
        Lreg_modes = torch.tensor(0.0, device=device)

    # Summen über Lastfälle (multi-load Training)
    Lphys_sum = torch.tensor(0.0, device=device)
    Lweak_sum = torch.tensor(0.0, device=device)
    Ldata_sum = torch.tensor(0.0, device=device)
    Lbc_sum   = torch.tensor(0.0, device=device)
    Lw_smooth_sum = torch.tensor(0.0, device=device)

    # Loop über Lastfälle:
    # k = Index des Lastfalls, net_w = Netz für w_k
    for k, ((_, _, q_t), net_w) in enumerate(zip(load_cases, net_w_list)):

        """
        STRONG FORM (punktweise PDE-Residual)

        Physikalische Gleichung (Euler-Bernoulli-Balken mit variabler Biegesteifigkeit):
            d²/dx² ( EI(x) * d²w/dx² ) = q(x)

        Interpretation:
        - w(x)  : Durchbiegung
        - w''(x): Krümmung
        - M(x)  = EI(x) * w''(x)  ist das Biegemoment
        - d²M/dx² = q(x) ist die Gleichgewichtsbedingung

        Implementationsdetail (Kettenregel):
        Das w-Netz ist in der skalierten Koordinate x_hat ∈ [-1,1] definiert.
        Es gilt:
            x_hat = 2x/L - 1  =>  dx_hat/dx = 2/L (konstant)

        Ableitungen:
            dw/dx   = (dw/dx_hat) * (2/L)
            d²w/dx² = (d²w/dx_hat²) * (2/L)²

        Deshalb wird im Code gerechnet:
            w_xx = ((2/L)**2) * w_xhatxhat

        Warum erscheint hier effektiv eine "4. Ableitung"?
        - Formal enthält die DGL (EI * w'')''.
        - Bei konstantem EI wäre das proportional zu w''''.
        - Bei variablem EI ist es korrekt die zweite Ableitung des Moments zu bilden:
            M = EI * w''
            M_x  = dM/dx
            M_xx = d²M/dx²
        - Genau dieser Term wird hier mittels Autograd berechnet.

        Das Strong-Residual:
            r(x) = M_xx(x) - q(x)
        wird an zufälligen Collocationpunkten ausgewertet und im Mittel quadratisch minimiert.
        """

        # w_f ist w(x_hat) am Collocation-Set
        w_f = net_w(x_f_hat_t)

        # w''(x) via x_hat
        # Achtung: net_w hängt von x_hat ab, aber Physik ist in x formuliert.
        # Daher: Ableitung w.r.t. x_hat -> Umrechnung per Kettenregel:
        # x_hat = 2x/L - 1  => dx_hat/dx = 2/L
        w_xhat = grad(w_f, x_f_hat_t)
        w_xhatxhat = grad(w_xhat, x_f_hat_t)
        # w_xx = d²w/dx² = (dx_hat/dx)^2 * d²w/dx_hat²
        w_xx = ((2.0 / L) ** 2) * w_xhatxhat

        # Biegemoment M = EI * w''
        M = EI_f * w_xx

        # (EI*w'')'' = d²M/dx²
        # Hier werden Ableitungen w.r.t. physikalischem x genommen (x_f_t).
        # Das ist wichtig, weil EI(x) ebenfalls im physikalischen Raum definiert ist.
        M_x  = grad(M, x_f_t)
        M_xx = grad(M_x, x_f_t)

        # Last q(x) am Collocation-Set
        q_f = q_t(x_f_t)

        # PDE-Residual r = (EI*w'')'' - q
        r = M_xx - q_f
        Lphys_sum = Lphys_sum + torch.mean(r**2)

        # Weak Form Zusatzterm: R_j = ∫(EI*w''*v'' - q*v) dx
        # Numerische Integration wird hier durch Monte-Carlo/Mean über zufällige Punkte angenähert:
        #   ∫ f(x) dx ≈ L * mean(f(x_samples))
        if lam_weak > 0.0:
            for j in range(1, n_test + 1):
                v, v_xx = v_sin_and_vxx_sin(x_f_t, j)
                integrand = EI_f * w_xx * v_xx - q_f * v
                Rj = L * torch.mean(integrand)
                Lweak_sum = Lweak_sum + Rj**2

        # Datenloss:
        # Netzvorhersage an Messpunkten x_d_t (in x_hat)
        w_pred = net_w(x_d_t)
        Ldata_sum = Ldata_sum + torch.mean((w_pred - w_d_t_list[k])**2)

        # Randbedingungen: w=0 und M=EI*w''=0 an x=0,L
        # w wird über das Netz an den Randpunkten (x_hat=-1 und +1) ausgewertet
        w0 = net_w(x0_hat_req)
        wL = net_w(xL_hat_req)

        # w'' am Rand (w.r.t. x) über Ableitungen w.r.t. x_hat (Kettenregel wie oben)
        w0_xhat = grad(w0, x0_hat_req)
        w0_xhatxhat = grad(w0_xhat, x0_hat_req)
        w0_xx = ((2.0 / L)**2) * w0_xhatxhat

        wL_xhat = grad(wL, xL_hat_req)
        wL_xhatxhat = grad(wL_xhat, xL_hat_req)
        wL_xx = ((2.0 / L)**2) * wL_xhatxhat

        # EI am Rand in physikalischem x (0 und L)
        EI0 = ei_model(x0)
        EIL = ei_model(xL)

        # Momentbedingung M=EI*w'' = 0
        M0 = EI0 * w0_xx
        ML = EIL * wL_xx

        # BC-Loss:
        # - w0,wL -> Durchbiegung an Lagern
        # - M0,ML -> Moment an Lagern
        Lbc_sum = Lbc_sum + (w0**2).mean() + (wL**2).mean() + (M0**2).mean() + (ML**2).mean()

        # optional: w-Glattheit (w'')^2
        # - wirkt wie eine Regularisierung der Krümmung von w
        if lam_w_smooth > 0.0:
            Lw_smooth_sum = Lw_smooth_sum + torch.mean(w_xx**2)

    # Gesamt-Loss als gewichtete Summe
    loss = (
        lam_phys * Lphys_sum
        + lam_weak * Lweak_sum
        + lam_data * Ldata_sum
        + lam_bc * Lbc_sum
        + lam_reg_ei * Lreg_ei
        + lam_w_smooth * Lw_smooth_sum
        + lam_modes * Lreg_modes
    )

    return loss, Lphys_sum, Lweak_sum, Ldata_sum, Lbc_sum, Lreg_ei, Lw_smooth_sum, Lreg_modes


def loss_all_weak(
    x_f_t, x_f_hat_t,
    lam_phys, lam_data, lam_bc,
    lam_reg_ei, lam_w_smooth,
    lam_modes,
    n_test=8
):
    # Variante für Phase C:
    # - hier wird "Physik" rein über Weak-Form abgebildet (Lphys_sum),
    #   Strong-Residual wird nicht benutzt.
    # - Das reduziert i.d.R. Probleme mit sehr hohen Ableitungen (M_xx etc.),
    #   kann aber auch "weicher" sein und hängt von Testfunktionen ab.

    # Randpunkte frisch (Graph sauber)
    x0_hat_req = x0_hat.clone().detach().requires_grad_(True)
    xL_hat_req = xL_hat.clone().detach().requires_grad_(True)

    EI_f = ei_model(x_f_t)

    # EI-Glattheit (optional)
    if lam_reg_ei > 0.0:
        EI_x = grad(EI_f, x_f_t)
        Lreg_ei = torch.mean(EI_x**2)
    else:
        Lreg_ei = torch.tensor(0.0, device=device)

    # Knotenkrümmung (optional)
    if lam_modes > 0.0 and ei_model.m >= 3:
        EI_k = ei_model.knot_values()
        d2 = EI_k[2:] - 2.0 * EI_k[1:-1] + EI_k[:-2]
        Lreg_modes = torch.mean(d2**2)
    else:
        Lreg_modes = torch.tensor(0.0, device=device)

    Lphys_sum = torch.tensor(0.0, device=device)
    Ldata_sum = torch.tensor(0.0, device=device)
    Lbc_sum   = torch.tensor(0.0, device=device)
    Lw_smooth_sum = torch.tensor(0.0, device=device)

    """
    WEAK FORM (integrale Residuen) – Randbedingungen in dieser Variante

    Grundidee der Weak Form:
    - Statt das PDE-Residual punktweise zu minimieren, werden integrierte Residuen
    gegen Testfunktionen v_j(x) minimiert:
        R_j = ∫_0^L ( EI(x)*w''(x)*v_j''(x) - q(x)*v_j(x) ) dx
    - Dies entspricht der klassischen Variationsformulierung des
    Euler-Bernoulli-Balkenproblems.

    Numerische Umsetzung:
    - Die Integrale werden über zufällige Collocationpunkte angenähert:
        ∫ f(x) dx ≈ L * mean( f(x_samples) )

    Warum werden hier nur w(0)=w(L)=0 erzwungen?
    - In dieser loss_all_weak-Variante werden bewusst nur die essenziellen
    Randbedingungen (Durchbiegung an den Lagern) penalisiert.
    - Die natürlichen Randbedingungen (Moment M=EI*w'' = 0) werden hier
    nicht explizit in den Loss aufgenommen.

    Motivation:
    - Die Weak-Form benötigt keine explizite Berechnung von M_xx
    (zweite Ableitung des Moments), was numerisch robuster ist.
    - Zusätzliche Momenten-Randbedingungen können den Loss stark versteifen
    und die L-BFGS-Optimierung instabiler machen.
    - Für die finale Feinoptimierung (Phase C) wird daher ein bewusst
    reduzierter, "balancierter" Physik-Loss verwendet.

    Hinweis:
    - Momenten-Randbedingungen könnten alternativ explizit ergänzt werden,
    wurden hier jedoch absichtlich weggelassen.
    """

    for k, ((_, _, q_t), net_w) in enumerate(zip(load_cases, net_w_list)):
        w_f = net_w(x_f_hat_t)

        # w''(x)
        w_xhat = grad(w_f, x_f_hat_t)
        w_xhatxhat = grad(w_xhat, x_f_hat_t)
        w_xx = ((2.0 / L) ** 2) * w_xhatxhat

        q_f = q_t(x_f_t)

        # Weak residuals
        for j in range(1, n_test + 1):
            v, v_xx = v_sin_and_vxx_sin(x_f_t, j)
            integrand = EI_f * w_xx * v_xx - q_f * v
            Rj = L * torch.mean(integrand)
            Lphys_sum = Lphys_sum + Rj**2

        # Datenloss
        w_pred = net_w(x_d_t)
        Ldata_sum = Ldata_sum + torch.mean((w_pred - w_d_t_list[k])**2)

        # BC: essential
        # In dieser Weak-only-Variante werden nur w=0-Randbedingungen erzwungen (keine Moment-BC).
        # Das ist eine Modellierungsentscheidung: Moment-BC könnten zusätzlich integriert werden,
        # sind hier aber bewusst nicht enthalten.
        w0 = net_w(x0_hat_req)
        wL = net_w(xL_hat_req)
        Lbc_sum = Lbc_sum + (w0**2).mean() + (wL**2).mean()

        if lam_w_smooth > 0.0:
            Lw_smooth_sum = Lw_smooth_sum + torch.mean(w_xx**2)

    loss = (
        lam_phys * Lphys_sum
        + lam_data * Ldata_sum
        + lam_bc * Lbc_sum
        + lam_reg_ei * Lreg_ei
        + lam_w_smooth * Lw_smooth_sum
        + lam_modes * Lreg_modes
    )
    return loss, Lphys_sum, Ldata_sum, Lbc_sum, Lreg_ei, Lw_smooth_sum, Lreg_modes

# ============================================================
# NEU: Plot-Ordner (eine Ebene hoch, dann "Plots")
# ============================================================
import os
PLOT_DIR = os.path.join("..", "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# NEU: Schleife über verschiedene Messrausch-Level
# ============================================================
base_seed = 0

for noise_pct in noise_levels_pct:
    print("\n\n====================================================")
    print(f"RUN mit Messrauschen = {noise_pct}%")
    print("====================================================")

    # Reproduzierbarkeit pro Run:
    # - Wir nutzen pro Noise-Level einen eigenen Seed, damit die Läufe deterministisch bleiben.
    rng = np.random.default_rng(base_seed + noise_pct)
    np.random.seed(base_seed + noise_pct)
    torch.manual_seed(base_seed + noise_pct)

    # Messwerte für alle Lastfälle:
    # w_ref_list: komplette Referenzlösung auf feinem Gitter
    # w_data_list: "gemessene" Werte nur an den Nd Punkten (plus optional Rauschen)
    w_ref_list, w_data_list = [], []
    for _, q_np, _ in load_cases:
        _, w_ref, _, _ = forward_solve_fd(q_np, N=801)
        w_ref_list.append(w_ref.copy())

        # NEU: Prozent-Rauschen (bezogen auf std der sauberen Messwerte)
        y_clean = w_ref[idx_d]
        sigma = (noise_pct / 100.0) * np.std(y_clean)
        y_noisy = y_clean + sigma * np.random.randn(Nd)
        w_data_list.append(y_noisy)

    # w-Daten für jeden Lastfall als Torch Tensor
    w_d_t_list = [torch.tensor(w.reshape(-1, 1), dtype=torch.float32, device=device) for w in w_data_list]

    # ============================================================
    # 4) Netze für w_k(x_hat)
    # ============================================================
    # Liste der w-Netze: eins pro Lastfall
    net_w_list = [MLP(1, 1, width=64, depth=4).to(device) for _ in load_cases]

    # ============================================================
    # 5) EI(x) als Control-Points (linear) + Positivität via Softplus
    # ============================================================
    # EI-Modell instanziieren:
    # m=21 -> 21 Knotenwerte, relativ flexible (aber noch glatte) Rekonstruktion möglich
    ei_model = EIControlPoints(m=21, beta=5.0, init_EI=1.0).to(device)     #konst: m=3     quad=Sin: m=21 init=1.0

    # ============================================================
    # 9) Training: Phase A + Phase B (Strong+Weak-Ramp) + Best-Tracking
    # ============================================================
    # Parameterlisten:
    # - params_w: alle Parameter aller w-Netze (pro Lastfall)
    # - params_all: EI-Parameter + w-Parameter (gemeinsames Training ab Phase B)
    params_w = []
    for nw in net_w_list:
        params_w += list(nw.parameters())
    params_all = list(ei_model.parameters()) + params_w

    # --------------------------
    # Phase A: w an Daten (EI eingefroren)
    # --------------------------
    # Idee Phase A:
    # - EI ist zunächst nicht lernbar (frozen), damit w-Netze sauber auf die Daten fitten.
    # - Dadurch bekommt man eine "gute" Initialisierung der w-Netze, bevor man die inverse Aufgabe
    #   (EI-Rekonstruktion) voll freischaltet.
    epochs_A = 2500

    # Starke Gewichtung der Daten und Randbedingungen in Phase A
    lam_data_A = 5e4
    lam_bc_A   = 5e4

    # Physikterm in Phase A ausgeschaltet (0.0), weil EI noch falsch/ungewiss ist
    lam_phys_A = 0.0

    # Keine Regularisierung in Phase A
    lam_reg_A  = 0.0
    lam_w_smooth_A = 0.0

    # EI einfrieren: gradients aus, EI bleibt konstant auf Init
    for p in ei_model.parameters():
        p.requires_grad_(False)

    # Adam-Optimizer nur für w-Netze
    optA = optim.Adam(params_w, lr=1e-3)

    print("\n--- Phase A (nur Daten+BC; EI eingefroren) ---")
    t_adam_start = time.perf_counter()

    for ep in range(1, epochs_A + 1):
        # In Phase A werden Collocationpunkte nur "formell" gebaut,
        # weil loss_all die Struktur einheitlich erwartet.
        # Physikgewicht ist 0, somit beeinflussen die Collocationpunkte die Optimierung nicht.
        x_f_t, x_f_hat_t = build_collocation(Nf=512)

        # Gradients zurücksetzen
        optA.zero_grad(set_to_none=True)

        # Loss berechnen (hier nur Daten + BC aktiv)
        loss, Lp, Lweak, Ld, Lbc, Lreg_ei, Lw_smooth, Lmodes = loss_all(
            x_f_t, x_f_hat_t,
            lam_phys=lam_phys_A, lam_data=lam_data_A, lam_bc=lam_bc_A,
            lam_reg_ei=lam_reg_A, lam_w_smooth=lam_w_smooth_A,
            lam_modes=0.0,
            lam_weak=0.0,
            n_test=8
        )

        # Backpropagation nur durch w-Netze (EI ist eingefroren)
        loss.backward()
        optA.step()

        # Logging alle 250 Epochen
        if ep % 250 == 0:
            print(f"A ep {ep:4d} | loss {loss.item():.3e} | Ldata {Ld.item():.3e} | "
                  f"Lbc {Lbc.item():.3e} | Lw_smooth {Lw_smooth.item():.3e}")

    # --------------------------
    # Phase B: EI + w gemeinsam (Strong+Weak-Ramp)
    # --------------------------
    # Idee Phase B:
    # - EI wird wieder trainierbar gemacht
    # - Physikterm wird langsam "angerampt" (von 0 auf lam_phys_cap)
    #   damit das Training nicht abrupt instabil wird.
    # - Zusätzlich: Weak-Form-Term ebenfalls rampen (oft stabilisierend / ergänzend).
    for p in ei_model.parameters():
        p.requires_grad_(True)

    # Ramp-Parameter:
    # - ramp_epochs: nach dieser Anzahl Epochen ist lam_phys_B = lam_phys_cap
    # - weak_ramp_epochs: Weak-Term ramp langsamer (hier 3000)
    ramp_epochs = 1000          #konst: 2000    Sin: 2000   quad: 1000
    epochs_B = 6000           #bei Sinus und quadratisch reicht 6000, bei konst ruhig auf 7000, da keine Phase C

    # Gewichte in Phase B:
    # - Daten immer noch hoch (um Messdaten einzuhalten)
    # - BC geringer als in Phase A (da Physik jetzt auch "mitreden" soll)
    # - EI-Regularisierung und Knoten-Glättung aktiv
    lam_data_B = 6e4            #konst=sin=quad: 60000.0
    lam_bc_B   = 1e3            #konst: 1000.0  Sin: 2000.0     quad: 1000.0
    lam_reg_B  = 0.01        # (dEI/dx)^2   konst: 10.0     quad=sin: 0.01
    lam_w_smooth_B = 0     # meist aus    Sin: 1e-6
    lam_modes_B = 1.0        # Knoten-Glättung  konst: 100.0    quad=sin: 1.0

    # NEU: Weak in Phase B zusätzlich
    lam_phys_cap = 50         #konst: 7.5     Sin:40   quad: 50.0
    lam_weak_cap = 5.0          # typisch 0.5..20 (je nach Skalierung)   konst: 0.0    Sin: 0.5     quad: 5.0
    weak_ramp_epochs = 3000     # langsamer rampen  konst=quad: 3000    Sin: 3500
    n_test_B = 8

    # Adam auf allen Parametern (EI + alle w-Netze)
    optB = optim.Adam(params_all, lr=1e-3)

    print("\n--- Phase B (EI+w, Strong+Weak-Ramp) ---")

    # Best-Tracking:
    # - best_any_*: bestes Modell über alle Epochen (egal ob lam_phys schon "voll" ist)
    # - best_cap_*: bestes Modell NACHDEM lam_phys auf dem Cap angekommen ist (volle Physik)
    best_cap_loss = float("inf")
    best_cap_state = None
    best_cap_epoch = None

    best_any_loss = float("inf")
    best_any_state = None
    best_any_epoch = None

    for ep in range(1, epochs_B + 1):
        # Ramp von lam_phys:
        # - startet bei klein und wächst linear bis lam_phys_cap
        lam_phys_B = lam_phys_cap * min(1.0, ep / ramp_epochs)

        # Ramp von lam_weak:
        # - wächst langsamer bis lam_weak_cap
        lam_weak_B = lam_weak_cap * min(1.0, ep / weak_ramp_epochs)

        # Collocationpunkte (jetzt relevant, weil Physik/Weak aktiv sind)
        x_f_t, x_f_hat_t = build_collocation(Nf=1024)

        optB.zero_grad(set_to_none=True)

        # Loss (Strong + Weak + Data + BC + EI-Glattheit + Knoten-Glättung)
        loss, Lp, Lweak, Ld, Lbc, Lreg_ei, Lw_smooth, Lmodes = loss_all(
            x_f_t, x_f_hat_t,
            lam_phys=lam_phys_B, lam_data=lam_data_B, lam_bc=lam_bc_B,
            lam_reg_ei=lam_reg_B, lam_w_smooth=lam_w_smooth_B,
            lam_modes=lam_modes_B,
            lam_weak=lam_weak_B,
            n_test=n_test_B
        )

        loss.backward()
        optB.step()

        loss_val = float(loss.item())

        # Speichern des besten Zustands (best_any)
        # Wichtig: detach().cpu().clone() um:
        # - Graph zu trennen
        # - Modellzustand unabhängig vom Device/aktuellen Tensor zu speichern
        if loss_val < best_any_loss:
            best_any_loss = loss_val
            best_any_epoch = ep
            best_any_state = {
                "ei": {k: v.detach().cpu().clone() for k, v in ei_model.state_dict().items()},
                "w":  [{k: v.detach().cpu().clone() for k, v in nw.state_dict().items()} for nw in net_w_list],
                "loss": loss_val,
                "epoch": ep,
                "lam_phys": float(lam_phys_B),
                "lam_weak": float(lam_weak_B),
            }

        # Speichern des besten Zustands unter "voller Physik" (best_cap)
        if lam_phys_B >= lam_phys_cap - 1e-12:
            if loss_val < best_cap_loss:
                best_cap_loss = loss_val
                best_cap_epoch = ep
                best_cap_state = {
                    "ei": {k: v.detach().cpu().clone() for k, v in ei_model.state_dict().items()},
                    "w":  [{k: v.detach().cpu().clone() for k, v in nw.state_dict().items()} for nw in net_w_list],
                    "loss": loss_val,
                    "epoch": ep,
                    "lam_phys": float(lam_phys_B),
                    "lam_weak": float(lam_weak_B),
                }

        # Logging alle 250 Epochen
        if ep % 250 == 0:
            # EI an der Balkenmitte als schnelle Plausibilitätszahl
            with torch.no_grad():
                EI_mid = ei_model(torch.tensor([[0.5 * L]], dtype=torch.float32, device=device)).item()

            # Beiträge der einzelnen Loss-Terme (gewichtete Komponenten),
            # damit man sieht, welcher Term dominiert.
            contrib_phys = lam_phys_B * Lp.item()
            contrib_weak = lam_weak_B * Lweak.item()
            contrib_data = lam_data_B * Ld.item()
            contrib_bc   = lam_bc_B   * Lbc.item()

            # Info zum gespeicherten Best-State
            if best_cap_state is not None:
                msg_best = f" | best_cap ep {best_cap_epoch} loss {best_cap_loss:.3e}"
            else:
                msg_best = f" | best_any ep {best_any_epoch} loss {best_any_loss:.3e}"

            print(f"B ep {ep:4d} | loss {loss.item():.3e} | "
                  f"Lphys {Lp.item():.3e} | Lweak {Lweak.item():.3e} | "
                  f"Ldata {Ld.item():.3e} | Lbc(M) {Lbc.item():.3e} | "
                  f"EI_mid {EI_mid:.4f} | lam_phys {lam_phys_B:.1f} | lam_weak {lam_weak_B:.2f} || "
                  f"contrib phys {contrib_phys:.3e} | weak {contrib_weak:.3e} | "
                  f"data {contrib_data:.3e} | bc {contrib_bc:.3e}"
                  f"{msg_best}")

    t_adam_end = time.perf_counter()

    # Reload best model after Phase B
    # Motivation:
    # - Nach dem Training laden wir den besten gespeicherten Zustand (nicht zwingend den letzten).
    # - Falls best_cap_state existiert, bevorzugen wir das Modell, das unter voller Physik am besten war.
    state_to_load = best_cap_state if best_cap_state is not None else best_any_state
    if state_to_load is None:
        print("WARNUNG: Kein Best-State gespeichert (sollte nicht passieren).")
    else:
        print("\n--- Best-Model Reload (nach Phase B) ---")
        print(f"Loading best model from epoch {state_to_load['epoch']} with loss {state_to_load['loss']:.6e} "
              f"(lam_phys={state_to_load['lam_phys']:.2f}, lam_weak={state_to_load['lam_weak']:.2f})")

        # Zustände zurück in die Modelle laden
        ei_model.load_state_dict(state_to_load["ei"])
        ei_model.to(device)
        for nw, sd in zip(net_w_list, state_to_load["w"]):
            nw.load_state_dict(sd)
            nw.to(device)

    # ============================================================
    # 10) Phase C: L-BFGS (Balanced)
    # ============================================================
    # Phase C nutzt L-BFGS als Second-Order-ähnlichen Optimierer.
    # Typische Idee bei PINNs:
    # - erst Adam (robust, explorativ)
    # - dann L-BFGS (fein, schnell in lokale Minima)
    #
    # Hier wird eine "balanced" Loss-Konfiguration genutzt, und die Physik wird über Weak-Form formuliert.

    USE_Phase_C = True
    if USE_Phase_C:
        print("\n--- Phase C (L-BFGS Finetune, Balanced) ---")

        lbfgs = optim.LBFGS(
            params_all,
            lr=1.0,
            max_iter=1000,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        # Fixe Collocation-Punkte:
        # - Bei L-BFGS ist es üblich (und oft stabiler), die Kollokationspunkte nicht ständig zu ändern,
        #   weil L-BFGS aus vergangenen Gradienteninformationen eine Approximation der Hesse-Matrix aufbaut.
        x_f_t_fix, _ = build_collocation(Nf=8196)

        lam_data_final = 30000.0            #Sin: 30000.0    quad: 20000.0
        lam_phys_final = 0.3                #Sin: 0.3        quad: 1.0
        lam_bc_final   = 10000.0             #Sin: 10000.0    quad: 1000.0
        lam_modes_final = 1.0               #Sin=konst=quad: 1.0
        lam_w_smooth_final = 0.0         #Sin: 0.0        quad: 1e-4

        # Vorher-Check
        # - einmal Loss auswerten, bevor L-BFGS startet
        x_hat_temp = x_hat_t(x_f_t_fix)
        loss_before, Lp_b, Ld_b, Lbc_b, *_ = loss_all_weak(
            x_f_t_fix, x_hat_temp,
            lam_phys=lam_phys_final, lam_data=lam_data_final, lam_bc=lam_bc_final,
            lam_reg_ei=0.0, lam_w_smooth=lam_w_smooth_final, lam_modes=lam_modes_final,
            n_test=8
        )
        print(f"LBFGS BEFORE | Loss {loss_before.item():.4e}")

        # NEU: Counter ohne nonlocal/global (robust)
        it_c = {"val": 0}

        def closure():
            # Closure ist Pflicht bei PyTorch-LBFGS:
            # - L-BFGS ruft die closure mehrfach pro Iteration auf (Line Search / Wolfe conditions)
            # - closure muss: gradients nullen, loss berechnen, backward ausführen, loss zurückgeben
            lbfgs.zero_grad(set_to_none=True)

            # x_hat muss frisch aus x_f_t_fix berechnet werden
            # (hier ist x_f_t_fix konstant, aber man erzeugt eine saubere Ableitungskette)
            x_f_hat_t_fresh = x_hat_t(x_f_t_fix)

            loss, Lp_c, Ld_c, Lbc_c, Lreg_c, Lw_smooth_c, Lmodes_c = loss_all_weak(
                x_f_t_fix,
                x_f_hat_t_fresh,
                lam_phys=lam_phys_final,
                lam_data=lam_data_final,
                lam_bc=lam_bc_final,
                lam_reg_ei=0.0,
                lam_w_smooth=lam_w_smooth_final,
                lam_modes=lam_modes_final,
                n_test=8
            )

            # Backprop für L-BFGS
            loss.backward()

            # Logging alle 50 closure-Aufrufe (nicht identisch zu "Iterationen" im Sinne von Epochen)
            it_c["val"] += 1
            if it_c["val"] % 50 == 0:
                print(
                    f"LBFGS iter {it_c['val']:4d} | "
                    f"loss {loss.item():.3e} | "
                    f"Lphys {Lp_c.item():.3e} | "
                    f"Lw_smooth {Lw_smooth_c.item():.3e} | "
                    f"contrib w_smooth {(lam_w_smooth_final * Lw_smooth_c.item()):.3e}"
                )

            return loss

        # L-BFGS Optimierung startet hier
        lbfgs.step(closure)

        # Nachher-Check
        x_hat_temp_after = x_hat_t(x_f_t_fix)
        loss_after, *_ = loss_all_weak(
            x_f_t_fix, x_hat_temp_after,
            lam_phys=lam_phys_final, lam_data=lam_data_final, lam_bc=lam_bc_final,
            lam_reg_ei=0.0, lam_w_smooth=lam_w_smooth_final, lam_modes=lam_modes_final,
            n_test=8
        )
        print(f"LBFGS AFTER  | Loss {loss_after.item():.4e}")

    # ============================================================
    # 11) Auswertung
    # ============================================================
    # Hier werden die rekonstruierten Größen auf dem Referenzgitter ausgewertet:
    # - EI_hat(x): rekonstruiertes EI
    # - w_hat_list: rekonstruiertes w für jede Lastform
    with torch.no_grad():
        x_plot = torch.tensor(x_ref.reshape(-1, 1), dtype=torch.float32, device=device)
        x_plot_hat = x_hat_t(x_plot)

        EI_hat_t = ei_model(x_plot)
        EI_hat = EI_hat_t.detach().cpu().numpy().reshape(-1)

        w_hat_list = [nw(x_plot_hat).detach().cpu().numpy().reshape(-1) for nw in net_w_list]

    # ============================================================
    # FD-Forward mit EI_PINN
    # ============================================================
    w_fd_from_EIhat_list = []
    for _, q_np, _ in load_cases:
        # Achtung: wir nutzen das gleiche FD-Gitter (x_ref) => N muss matchen
        _, w_fd_from_EIhat, _, _ = forward_solve_fd_with_EI(EI_hat, q_np, N=len(x_ref))
        w_fd_from_EIhat_list.append(w_fd_from_EIhat.copy())

    # ============================================================
    # Kennzahlen: Vergleich w_ref vs w_FD(EI_PINN)
    # ============================================================
    w_fd_l2_rel_list = []
    w_fd_max_abs_list = []

    for k, (name, _, _) in enumerate(load_cases):
        w_ref = w_ref_list[k].reshape(-1)
        w_fd  = w_fd_from_EIhat_list[k].reshape(-1)

        rel_l2 = np.linalg.norm(w_fd - w_ref) / (np.linalg.norm(w_ref) + 1e-12)
        max_abs = np.max(np.abs(w_fd - w_ref))

        w_fd_l2_rel_list.append(rel_l2)
        w_fd_max_abs_list.append(max_abs)

        print(f"[FD mit EI_PINN] Lastfall {name:12s} | relL2(w) = {rel_l2:.3e} | max|Δw| = {max_abs:.3e}")

    # Relativer L2-Fehler der EI-Rekonstruktion:
    #   ||EI_hat - EI_ref|| / ||EI_ref||
    EI_l2_rel = np.linalg.norm(EI_hat - EI_ref) / (np.linalg.norm(EI_ref) + 1e-12)
    print(f"\nRelativer L2-Fehler EI: {EI_l2_rel:.3e}")

    # Laufzeiten ausgeben
    t_total_end_run = time.perf_counter()
    print(f"Adam-Zeit:   {t_adam_end - t_adam_start:.2f} s")
    print(f"Run-Zeit:    {t_total_end_run - t_total_start:.2f} s")

    # ============================================================
    # 12) Plots
    # ============================================================
    # Plotten:
    # - EI_true vs EI_PINN
    # - w_ref, w_data, w_PINN pro Lastfall
    
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x_ref, EI_ref, label=f"EI_Referenz ({ei_verlauf})")
        plt.plot(x_ref, EI_hat, "--", label=f"EI_PINN (ControlPoints m={ei_model.m})")
        plt.xlabel("x")
        plt.ylabel("EI")
        plt.title(f"Biegesteifigkeit | EI via Control Points | Rauschen={noise_pct}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"EI_m35_{ei_verlauf}_noise{noise_pct}.png"), dpi=300)
        plt.close()

        for k, (name, _, _) in enumerate(load_cases):
            plt.figure()
            plt.plot(x_ref, w_ref_list[k], label="w_Referenz")
            plt.scatter(x_data, w_data_list[k], s=12, label="w_Daten")
            plt.plot(x_ref, w_hat_list[k], "--", label="w_PINN")
            plt.plot(x_ref, w_fd_from_EIhat_list[k], ":", label="w_EI_PINN")
            plt.xlabel("x")
            plt.ylabel("w")
            plt.title(f"Durchbiegung (Lastform: {name}) | Rauschen={noise_pct}%")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"w_{name}_EI_m35_{ei_verlauf}_noise_{noise_pct}.png"), dpi=300)
            plt.close()

    except ImportError:
        print("matplotlib nicht installiert – Plot wird übersprungen.")

# Laufzeiten ausgeben
t_total_end = time.perf_counter()
print(f"Gesamtzeit:  {t_total_end - t_total_start:.2f} s")