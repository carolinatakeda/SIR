import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import solve_ivp

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title="Modelo SIR Não Autônomo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Funções auxiliares
# -----------------------------
def analyze_mean(gamma0, gammaAmp, a, b, c):
    m_gamma = gamma0
    gamma_min = gamma0 - gammaAmp
    gamma_max = gamma0 + gammaAmp
    threshold = a + b + c
    condition_survive = m_gamma < threshold
    
    return {
        "m_gamma": round(m_gamma, 4),
        "gamma_min": round(gamma_min, 4),
        "gamma_max": round(gamma_max, 4),
        "threshold": round(threshold, 4),
        "condition": condition_survive,
        "condition_return": "m(γ) < a + b + c" if condition_survive else "m(γ) ≥ a + b + c"
    }

def sir_system(t, y, gamma_func, q_func, a, b, c):
    S, I, R = y
    N = S + I + R
    
    if N == 0:
        return np.array([0.0, 0.0, 0.0])
    
    gamma = gamma_func(t)
    q = q_func(t)
    
    dS = q - gamma * S * I / N - a * S + b * R
    dI = gamma * S * I / N - c * I - a * I
    dR = c * I - a * R - b * R
    
    return np.array([dS, dI, dR])

def N_star(t, q0, qAmp, a):
    """Calcula N*(t) - atrator pullback"""
    if a == 0:
        return q0 * t
    return q0 / a + qAmp * (a * np.sin(t) - np.cos(t)) / (a**2 + 1)

def run_simulation(a, b, c, gamma0, gammaAmp, q0, qAmp, S0, I0, R0, s, tMax):
    # Funções gamma(t) e q(t) periódicas
    def gamma_t(t):
        return gamma0 + gammaAmp * np.sin(t)
    
    def q_t(t):
        return q0 + qAmp * np.sin(t)
    
    def funcao_para_solve_ivp(t, y):
        return sir_system(t, y, gamma_t, q_t, a, b, c)
    
    # Condições iniciais
    y0 = np.array([S0, I0, R0], dtype=float)
    
    # Pontos de tempo para avaliar a solução (de s até s+tMax)
    t_eval = np.linspace(s, s + tMax, int(tMax * 10) + 1)
    
    sol = solve_ivp(
        fun=funcao_para_solve_ivp, 
        t_span=(s, s + tMax),
        y0=y0,
        method='RK45', 
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-6,
        atol=1e-9
    )
    
    # Verificar se a integração foi bem-sucedida
    if not sol.success:
        st.error(f"Erro na integração: {sol.message}")
        return pd.DataFrame()
    
    # Organizar resultados em DataFrame
    results = []
    for i, t in enumerate(sol.t):
        S, I, R = sol.y[:, i]
        N = S + I + R
        results.append({
            "t": round(t, 2),
            "S": S,
            "I": I,
            "R": R,
            "N": N,
            "N_star": N_star(t, q0, qAmp, a),
            "gamma": gamma_t(t),
            "q": q_t(t)
        })
    
    return pd.DataFrame(results)

def create_style_plot(data, color_S, color_I, color_R, title="", show_attractor=False):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
    
    # Painel 1: Suscetíveis
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['t'], data['S'], color=color_S, linewidth=2.5)
    ax1.set_ylabel('Suscetíveis', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(data['t'].min(), data['t'].max())
    
    # Painel 2: Infectados
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data['t'], data['I'], color=color_I, linewidth=2.5)
    ax2.set_ylabel('Infectados', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(data['t'].min(), data['t'].max())
    
    # Painel 3: Recuperados
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(data['t'], data['R'], color=color_R, linewidth=2.5)
    ax3.set_xlabel('Tempo (dias)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Recuperados', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(data['t'].min(), data['t'].max())
    
    # Painel grande: Diagrama de fase 3D
    ax4 = fig.add_subplot(gs[:, 1], projection='3d')
    
    # Colormap baseado no tempo
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    # Trajetória do sistema
    for i in range(len(data) - 1):
        ax4.plot(data['S'].iloc[i:i+2], 
                data['I'].iloc[i:i+2], 
                data['R'].iloc[i:i+2],
                color=colors[i], linewidth=2, alpha=0.7)
    
    # Se a condição for verdadeira, mostrar o atrator N*(t) no eixo S
    if show_attractor:
        # Trajetória do atrator: (N*(t), 0, 0)
        ax4.plot(data['N_star'], 
                np.zeros(len(data)), 
                np.zeros(len(data)),
                color='red', linewidth=3, linestyle='--', 
                alpha=0.8, label='Atrator: (N*, 0, 0)', zorder=3)
    
    # Ponto inicial
    ax4.scatter([data['S'].iloc[0]], [data['I'].iloc[0]], [data['R'].iloc[0]],
               color='green', s=150, marker='o', 
               label='Condição Inicial', zorder=5, edgecolor='black', linewidth=2)
    
    # Ponto final
    ax4.scatter([data['S'].iloc[-1]], [data['I'].iloc[-1]], [data['R'].iloc[-1]],
               color='red', s=150, marker='*', 
               label='Estado Final', zorder=5, edgecolor='black', linewidth=2)
    
    ax4.set_xlabel('Suscetíveis (S)', fontsize=12, fontweight='bold', labelpad=10)
    ax4.set_ylabel('Infectados (I)', fontsize=12, fontweight='bold', labelpad=10)
    ax4.set_zlabel('Recuperados (R)', fontsize=12, fontweight='bold', labelpad=10)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.view_init(elev=20, azim=45)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Ajuste manual do layout (tight_layout não funciona bem com 3D)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    return fig

# -----------------------------
# Inicializar session_state
# -----------------------------
if 'params' not in st.session_state:
    st.session_state.params = {
        'gamma0': 0.3, 'gammaAmp': 0.1, 'c': 0.1,
        'a': 0.02, 'b': 0.05, 'q0': 20.0, 'qAmp': 5.0,
        'S0': 900, 'I0': 50, 'R0': 50, 's': 0, 'tMax': 300
    }

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("🦠 Modelo SIR Não Autônomo")

# -----------------------------
# Barra lateral
# -----------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros do Modelo")
    
    # Presets no topo
    st.subheader("📋 Cenários Pré-configurados")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Doença\nExtinta", use_container_width=True):
            st.session_state.params = {
                'gamma0': 0.06, 'gammaAmp': 0.05, 'c': 0.1,
                'a': 0.02, 'b': 0.05, 'q0': 20.0, 'qAmp': 5.0,
                'S0': 900, 'I0': 50, 'R0': 50, 's': 0, 'tMax': 500
            }
    with col2:
        if st.button("🔴 Doença\nEndêmica", use_container_width=True):
            st.session_state.params = {
                'gamma0': 0.86, 'gammaAmp': 0.2, 'c': 0.1,
                'a': 0.02, 'b': 0.05, 'q0': 20.0, 'qAmp': 5.0,
                'S0': 900, 'I0': 50, 'R0': 50, 's': 0, 'tMax': 500
            }
    
    st.divider()
    
    with st.expander("🔬 Parâmetros Epidemiológicos", expanded=True):
        gamma0 = st.slider("γ₀ — taxa base de infecção", 0.0, 1.0, 
                          st.session_state.params['gamma0'], 0.01)
        gammaAmp = st.slider("Amplitude de γ(t)", 0.0, 0.5, 
                            st.session_state.params['gammaAmp'], 0.01)
        c = st.slider("c — taxa de recuperação", 0.01, 0.5, 
                     st.session_state.params['c'], 0.01)
        st.caption(f"Tempo médio infectado: {1/c:.1f} dias")
    
    with st.expander("👥 Parâmetros Demográficos", expanded=True):
        a = st.slider("a — taxa de mortalidade/emigração", 0.0, 0.1, 
                     st.session_state.params['a'], 0.001)
        b = st.slider("b — taxa de reinfecção", 0.0, 0.2, 
                     st.session_state.params['b'], 0.01)
        q0 = st.slider("q₀ — taxa base de nascimentos", 0.0, 50.0, 
                      st.session_state.params['q0'], 1.0)
        qAmp = st.slider("Amplitude de q(t)", 0.0, 20.0, 
                        st.session_state.params['qAmp'], 1.0)
        
        # Validação biológica: q(t) ≥ 0
        if qAmp > q0:
            st.warning(f"⚠️ **Restrição biológica violada!**\n\nPara garantir q(t) ≥ 0, é necessário: q₀ ≥ |A_q|\n\nAtual: q₀ = {q0:.1f} < A_q = {qAmp:.1f}")
            st.info(f"💡 Aumente q₀ para pelo menos {qAmp:.1f} ou reduza A_q para no máximo {q0:.1f}")
        else:
            q_min = q0 - qAmp
            q_max = q0 + qAmp
            st.success(f"✅ q(t) ∈ [{q_min:.1f}, {q_max:.1f}] (sempre ≥ 0)")
    
    with st.expander("🎯 Condições Iniciais", expanded=True):
        S0 = st.number_input("S₀ — suscetíveis iniciais", 0, 5000, 
                            st.session_state.params['S0'], 50)
        I0 = st.number_input("I₀ — infectados iniciais", 0, 500, 
                            st.session_state.params['I0'], 5)
        R0 = st.number_input("R₀ — recuperados iniciais", 0, 500, 
                            st.session_state.params['R0'], 10)
        st.caption(f"População inicial N₀ = {S0 + I0 + R0}")
    
    with st.expander("⏱️ Simulação", expanded=True):
        s = st.number_input("s — instante inicial", 
                           min_value=-1000.0, 
                           max_value=1000.0,
                           value=float(st.session_state.params['s']),
                           step=1.0,
                           help="Tempo em que a simulação começa")
        tMax = st.slider("Tempo máximo (dias)", 50, 100000, 
                        st.session_state.params['tMax'], 50)
    
    with st.expander("🎨 Opções Gráficas"):
        color_S = st.color_picker("Cor S (Suscetíveis)", "#1f77b4")
        color_I = st.color_picker("Cor I (Infectados)", "#d62728")
        color_R = st.color_picker("Cor R (Recuperados)", "#2ca02c")

# -----------------------------
# Executar simulação
# -----------------------------
with st.spinner("Simulando modelo não autônomo com scipy.integrate.solve_ivp..."):
    try:
        data = run_simulation(a, b, c, gamma0, gammaAmp, q0, qAmp, S0, I0, R0, s, tMax)
        
        if data.empty:
            st.error("Erro na simulação. Verifique os parâmetros.")
            st.stop()
            
        analysis = analyze_mean(gamma0, gammaAmp, a, b, c)
        
    except Exception as e:
        st.error(f"Erro durante a simulação: {str(e)}")
        st.stop()

# -----------------------------
# Análise dos dados
# -----------------------------
st.header("📊 Dados")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("m(γ)", analysis["m_gamma"], 
              help="Média de γ(t)")
with col2:
    st.metric("a + b + c", analysis["threshold"])
with col3:
    delta = analysis["m_gamma"] - analysis["threshold"]
    st.metric("Diferença", f"{delta:.4f}",
              delta=delta,
              help="m(γ) - (a+b+c)")

# Box com resultado do teorema
if analysis["condition"]:
    st.success(f"""
    **✅ Condição: {analysis['condition_return']}**
    
    **Teorema (Caso 1):** A doença é erradicada!
    
    - lim I(t,s,x₀) = 0 quando t → +∞
    - lim R(t,s,x₀) = 0 quando t → +∞
    - O atrator pullback é 𝒜(t) = {{(N*(t), 0, 0)}}
    - Sistema converge para estado livre de doença
    """)
else:
    st.error(f"""
    **⚠️ Condição: {analysis['condition_return']}**
    
    **Teorema (Caso 2):** A doença persiste endemicamente!
    
    - Existe ε₀ > 0 tal que lim I(t,s,x₀) > ε₀
    - Os infectados não tendem a zero
    - População mantém nível endêmico de infecção
    """)

# Sistema de equações
with st.expander("📐 Ver Sistema de Equações Não Autônomo"):
    st.latex(r"""
    \begin{cases}
    \frac{dS}{dt} = q(t) - \gamma(t) \frac{SI}{N} - aS + bR \\[0.5em]
    \frac{dI}{dt} = \gamma(t) \frac{SI}{N} - cI - aI \\[0.5em]
    \frac{dR}{dt} = cI - aR - bR
    \end{cases}
    """)
    
    st.markdown("**Funções não autônomas:**")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"\gamma(t) = \gamma_0 + A_\gamma \sin(t)")
        st.caption("Taxa de infecção")
    with col2:
        st.latex(r"q(t) = q_0 + A_q \sin(t)")
        st.caption("Nascimentos/imigrações")
    
    st.markdown("---")
    st.markdown("**Observação (Restrição Biológica):**")
    st.markdown("""
    A função $q(t) = q_0 + A_q\\sin(t)$ descreve a taxa de nascimento e imigração da população. 
    Por razões biológicas, essa taxa deve ser não-negativa para todo $t \\in \\mathbb{R}$. 
    Para garantir que $q(t) \\geq 0$ para todo $t$, precisamos impor a restrição:
    """)
    st.latex(r"q_0 \geq |A_q|")
    st.caption("Isso garante que q(t) nunca seja negativa, mantendo o modelo biologicamente plausível.")
    st.markdown("---")
    
    st.markdown("**Média da transmissão da doença:**")
    st.latex(r"""
    m(\gamma) := \limsup_{n \to +\infty} \left\{ \frac{1}{t - s} \int_s^t \gamma(\tau) \, d\tau \,:\, t - s \geq n \right\}
    """)
    st.caption("Para γ(t) periódica: m(γ) = γ₀")
    
    st.markdown("**Atrator pullback (caso estável):**")
    st.latex(r"N^*(t) = \int_{-\infty}^t e^{-a(t-r)} q(r) \, dr")
    st.markdown("Para $q(t) = q_0 + A_q\\sin(t)$, a solução exata é:")
    st.latex(r"N^*(t) = \frac{q_0}{a} + \frac{A_q(a\sin(t) - \cos(t))}{a^2 + 1}")
    st.caption("Caso particular: se A_q = 0 (q constante), então N*(t) = q₀/a")

# -----------------------------
# Gráfico principal
# -----------------------------
st.header("📈 Dinâmica SIR")

fig = create_style_plot(data, color_S, color_I, color_R, 
                       "Simulação SIR Não Autônomo",
                       show_attractor=analysis["condition"])
st.pyplot(fig)

if analysis["condition"]:
    st.info("""
    **Interpretação do gráfico:**
    - **Painéis esquerdos:** Evolução temporal de S(t), I(t) e R(t)
    - **Painel direito:** Diagrama de fase 3D mostrando trajetória no espaço (S, I, R)
    - **Ponto verde:** Condição inicial (S₀, I₀, R₀)
    - **Ponto vermelho:** Estado final após t_max dias
    - **Linha vermelha tracejada:** Atrator pullback (N*(t), 0, 0) - estado livre de doença
    - **Gradiente de cores:** Progressão temporal (roxo → amarelo)
    
    Note como a trajetória converge para o atrator no plano I=0, R=0.
    """)
else:
    st.info("""
    **Interpretação do gráfico:**
    - **Painéis esquerdos:** Evolução temporal de S(t), I(t) e R(t)
    - **Painel direito:** Diagrama de fase 3D mostrando trajetória no espaço (S, I, R)
    - **Ponto verde:** Condição inicial (S₀, I₀, R₀)
    - **Ponto vermelho:** Estado final após t_max dias
    - **Gradiente de cores:** Progressão temporal (roxo → amarelo)
    
    No regime endêmico, a trajetória não converge para um estado livre de doença.
    """)

# -----------------------------
# Gráficos das funções temporais
# -----------------------------
st.header("🔁 Funções Temporais γ(t) e q(t)")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(data['t'], data['gamma'], color='orange', linewidth=2.5, label='γ(t)')
    ax1.axhline(y=analysis["m_gamma"], color='purple', linestyle='-', 
                linewidth=2.5, label=f'm(γ) = {analysis["m_gamma"]} (média)')
    ax1.axhline(y=analysis["threshold"], color='green', linestyle='--', 
                linewidth=2, label=f'a+b+c = {analysis["threshold"]}')
    ax1.axhline(y=analysis["gamma_min"], color='red', linestyle=':', 
                linewidth=1.5, alpha=0.7, label=f'γ_mín = {analysis["gamma_min"]}')
    ax1.axhline(y=analysis["gamma_max"], color='blue', linestyle=':', 
                linewidth=1.5, alpha=0.7, label=f'γ_máx = {analysis["gamma_max"]}')
    ax1.fill_between(data['t'], 
                     analysis["gamma_min"], analysis["gamma_max"], 
                     alpha=0.1, color='orange')
    ax1.set_xlabel("Tempo (dias)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("γ(t)", fontsize=11, fontweight='bold')
    ax1.set_title("Taxa de Infecção γ(t)", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(data['t'], data['q'], color='teal', linewidth=2.5, label='q(t)')
    ax2.axhline(y=q0, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label=f'q₀ = {q0}')
    ax2.fill_between(data['t'], 
                     q0 - qAmp, q0 + qAmp, 
                     alpha=0.15, color='teal', label='Intervalo [q₀±A_q]')
    ax2.set_xlabel("Tempo (dias)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("q(t)", fontsize=11, fontweight='bold')
    ax2.set_title("Nascimentos/Imigrações q(t)", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

# -----------------------------
# População Total
# -----------------------------
st.header("👥 Dinâmica Populacional Total")

t = data['t']
N_star_values = data['N_star']

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(t, data['N'], label='N(t) — População Total', 
         color='darkblue', linewidth=2.5, alpha=0.8)

# Mostrar N*(t) apenas se a doença for erradicada
if analysis["condition"]:
    ax3.plot(t, N_star_values,  
             label='N*(t) — Atrator Pullback', color='red', linewidth=2, alpha=0.6, linestyle='--')

ax3.set_xlabel('Tempo (dias)', fontsize=11, fontweight='bold')
ax3.set_ylabel('População', fontsize=11, fontweight='bold')

if analysis["condition"]:
    ax3.set_title('População Total N(t) vs Atrator N*(t)', fontsize=13, fontweight='bold')
else:
    ax3.set_title('População Total N(t) — Regime Endêmico', fontsize=13, fontweight='bold')

ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

N_initial = data['N'].iloc[0]
N_final = data['N'].iloc[-1]
N_variation = abs(N_final - N_initial) / N_initial * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("N inicial", f"{N_initial:.1f}")
with col2:
    st.metric("N final", f"{N_final:.1f}", f"{N_final - N_initial:+.1f}")
with col3:
    st.metric("Variação", f"{N_variation:.2f}%")

# Mensagem explicativa diferente dependendo da condição
if analysis["condition"]:
    st.info("""
    **N*(t)** representa o atrator pullback (equilíbrio populacional livre de doença) quando m(γ) < a+b+c.
    
    Fórmula exata para q(t) = q₀ + A_q sin(t):
    
    N*(t) = q₀/a + A_q(a·sin(t) - cos(t))/(a² + 1)

    """)
else:
    st.warning("""
    **Regime Endêmico:** Como m(γ) ≥ a+b+c, a doença persiste na população e não há convergência 
    para o atrator pullback livre de doença. 
    """)

# -----------------------------
# Análise detalhada de I(t)
# -----------------------------
st.header("🔬 Análise Detalhada dos Infectados")

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

# I(t) linear
ax4a.plot(data['t'], data['I'], color=color_I, linewidth=2.5)
ax4a.set_xlabel("Tempo (dias)", fontsize=11, fontweight='bold')
ax4a.set_ylabel("I(t)", fontsize=11, fontweight='bold')
ax4a.set_title("Infectados - Escala Linear", fontsize=13, fontweight='bold')
ax4a.grid(True, alpha=0.3)

# I(t) log
I_positive = data['I'].replace(0, np.nan)
ax4b.plot(data['t'], I_positive, color=color_I, linewidth=2.5)
ax4b.set_xlabel("Tempo (dias)", fontsize=11, fontweight='bold')
ax4b.set_ylabel("I(t) (escala log)", fontsize=11, fontweight='bold')
ax4b.set_title("Infectados - Escala Logarítmica", fontsize=13, fontweight='bold')
ax4b.set_yscale('log')
ax4b.grid(True, alpha=0.3, which='both')

plt.tight_layout()
st.pyplot(fig4)

# Estatísticas finais
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("I inicial", f"{data['I'].iloc[0]:.2f}")
with col2:
    st.metric("I máximo", f"{data['I'].max():.2f}")
with col3:
    st.metric("I final", f"{data['I'].iloc[-1]:.4f}")
with col4:
    # Média dos últimos 10%
    last_10_percent = max(1, int(len(data) * 0.1))
    mean_final = data['I'].iloc[-last_10_percent:].mean()
    if mean_final < 1:
        st.success(f"✅ I→0\n({mean_final:.4f})")
    else:
        st.warning(f"⚠️ I>{mean_final:.2f}")

# -----------------------------
# Dados numéricos
# -----------------------------
with st.expander("📊 Ver Tabela de Dados"):
    st.dataframe(
        data[['t', 'S', 'I', 'R', 'N', 'N_star', 'gamma', 'q']].style.format({
            't': '{:.1f}',
            'S': '{:.2f}',
            'I': '{:.4f}',
            'R': '{:.2f}',
            'N': '{:.2f}',
            'N_star': '{:.2f}',
            'gamma': '{:.4f}',
            'q': '{:.2f}'
        }),
        height=400
    )
    
    # Download
    csv = data.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados em CSV",
        data=csv,
        file_name=f"sir_nonautonomous_mgamma_{analysis['m_gamma']}.csv",
        mime="text/csv"
    )


