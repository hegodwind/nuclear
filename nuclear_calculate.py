import json
import numpy as np
import re
import os  
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, eye, isspmatrix
from scipy.sparse.linalg import spsolve, expm as sparse_expm
from scipy.linalg import expm as dense_expm
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ==========================================
# 1. 后端逻辑 
# ==========================================

class NuclearLibrary:
    """
    核素库管理类：
    负责加载数据、解析物理关系，并将所有反应归类为：
    1. Decay (衰变)
    2. Reaction (中子俘获等)
    3. Fission (裂变产物生成)
    """
    def __init__(self, json_source=None):
        self.name_to_idx = {}   
        self.idx_to_name = []  
        
        # 衰变，反应，裂变事件列表
        self.decay_events = []      
        self.reaction_events = []   
        self.fission_events = []     
        
        # 拓扑关系
        self.capture_topology = {}
        self.decay_topology = {}

        if json_source:
            self.load_from_source(json_source)

    def _register_isotope(self, name):
        """注册新核素并返回索引"""
        if name not in self.name_to_idx:
            self.name_to_idx[name] = len(self.idx_to_name)
            self.idx_to_name.append(name)
        return self.name_to_idx[name]

    def load_from_source(self, source):
        if isinstance(source, str):
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif hasattr(source, 'read'): # File-like object
            data = json.load(source)
        elif isinstance(source, dict):
            data = source
        else:
            raise ValueError("Unsupported data source type")

        # 解析数据结构
        if "nuclides" in data:
            nuclides_data = data["nuclides"]
            self.capture_topology = data.get("capture_topology", {})
            self.decay_topology = data.get("decay_topology", {})
            
    
        # 第一步：注册所有核素
        for name in nuclides_data.keys():
            self._register_isotope(name)

        # 第二步：解析物理过程
        for name, info in nuclides_data.items():
            parent_idx = self.name_to_idx[name]
            
            # 1. 衰变 
            lam = info.get('decay_constant', 0.0)
            if lam > 0:
                child_name = self.decay_topology.get(name)
                child_idx = self.name_to_idx.get(child_name) if child_name else None
                self.decay_events.append((parent_idx, child_idx, lam))

            # 2. 中子反应
            sigma_c = info.get('sigma_capture', 0.0)
            if sigma_c > 0:
                child_name = self.capture_topology.get(name)
                child_idx = self.name_to_idx.get(child_name) if child_name else None
                self.reaction_events.append((parent_idx, child_idx, sigma_c))

            # 3. 裂变
            sigma_f = info.get('sigma_fission', 0.0)
            yields = info.get('fission_yields', {})
            
            if sigma_f > 0:
                # 记录裂变导致的父核素消失
                self.fission_events.append((parent_idx, None, (sigma_f, 0.0)))
                
                # 记录裂变产物的生成
                for fp_name, y_val in yields.items():
                    fp_idx = self.name_to_idx.get(fp_name)
                    if fp_idx is not None:
                        # 记录: (裂变源, 裂变产物, (截面, 产额))
                        self.fission_events.append((parent_idx, fp_idx, (sigma_f, y_val)))

    def get_matrix_size(self):
        return len(self.idx_to_name)


class BurnupMatrixFactory:
    """
    矩阵工厂：
    只负责数学计算。根据 Library 提供的数据和给定的通量 phi，
    组装稀疏矩阵 A。
    """
    def __init__(self, library):
        self.lib = library

    def build_matrix(self, phi,library):
        self.lib=library
        size = self.lib.get_matrix_size()
        A = lil_matrix((size, size))
        
        # 1. 处理衰变 
        for p_idx, c_idx, lam in self.lib.decay_events:
            A[p_idx, p_idx] -= lam  
            if c_idx is not None:
                A[c_idx, p_idx] += lam 

        # 2. 处理中子俘获 
        # sigma 单位转换: Barn -> cm^2
        unit_conv = 1e-24 
        
        for p_idx, c_idx, sigma in self.lib.reaction_events:
            rate = sigma * unit_conv * phi
            A[p_idx, p_idx] -= rate 
            if c_idx is not None:
                A[c_idx, p_idx] += rate 

        # 3. 处理裂变 
        for p_idx, c_idx, params in self.lib.fission_events:
            sigma_f, y_val = params
            
            if c_idx is None:
                rate = sigma_f * unit_conv * phi
                A[p_idx, p_idx] -= rate
            
            else:
                rate = sigma_f * y_val * unit_conv * phi
                A[c_idx, p_idx] += rate

        return A.tocsr()

#Cram方法所需的系数
cram_theta = np.array([
            -1.084391707869698026e+01 + 1.9277446167181652284e+01j,
            -5.2649713434426468895e+00 + 1.6220221473167927305e+01j,
            5.9481522689511774808e+00 + 3.5874573620183222829e+00j,
            3.5091036084149180974e+00 + 8.4361989858843750826e+00j,
            6.4161776990994341923e+00 + 1.194122393701386874e+00j,
            1.4193758971856659786e+00 + 1.0925363484496722585e+01j,
            4.993174737719963991e+00 + 5.996881713603942260e+00j,
            -1.4139284624888862114e+00 + 1.3497725698892745389e+01j
        ], dtype=complex)

        # 留数 alpha (Residues)
cram_alpha = np.array([
            -5.0901521865224915650e-07 - 2.4220017652852287970e-05j,
            2.115174218246030907e-04 + 4.3892969647380673918e-03j,
            1.1339775178483930527e+02 + 1.0194721704215856450e+02j,
            1.5059585270023467528e+01 -5.7514052776421819979e+00j,
            -6.4500878025539646595e+01 -2.2459440762652096056e+02j,
            -1.479300711355799971e+00 + 1.7686588323782937906e+00j,
            -6.2518392463207918892e+01 -1.1190391094283228480e+01j,
            4.1023136835410021273e-02 - 1.5743466173455468191e-01j
        ], dtype=complex)

        # 极限项 alpha0 (Limit at infinity)
cram_alpha0 = 2.124853710495223748e-016
        

class SimpleSolver:
    def __init__(self, method='scipy_expm', origen_order=15, origen_iter_limit=5):
        """
        method: 
          'scipy_expm': 使用 Scipy 自带的矩阵指数 (精度最高，最省事)
          'origen': 使用 ORIGEN 方法 (泰勒展开 + 短寿命核素平衡近似)
          'cram': 使用 CRAM 方法 (切比雪夫有理逼近法)
        """
        self.method = method
        self.origen_order = origen_order
        self.origen_iter_limit = origen_iter_limit

    def solve(self, A, N0, dt, steps=1):
        """
        计算 N(t+dt)
        """
        # 1. 准备工作
        size = A.shape[0]
        dt_sub = dt / steps
        current_N = N0.copy()
        
        # 2. 循环子步
        for s in range(steps):
            if self.method == 'scipy_expm':
                # --- 方法 A: 直接调用 Scipy ---
                # 构造 M = A * dt_sub
                M = A * dt_sub
                
                # 计算 exp(M)
                if isspmatrix(M):
                    propagator = sparse_expm(M)
                else:
                    propagator = dense_expm(M)
                
                # 更新浓度: N = exp(A*dt) * N
                current_N = propagator.dot(current_N)

            elif self.method == 'origen':
                # --- 方法 C: ORIGEN 方法 ---
                # 1. 区分长短寿命核素 
                # 判据: lambda_eff > -ln(0.001) / t
                limit_decay = -np.log(0.001) / dt_sub
                
                # 获取对角线元素 
                diag = A.diagonal()
                removal_rates = -diag
                
                # 找出短寿命核素的索引
                short_mask = removal_rates > limit_decay
                long_mask = ~short_mask
                
                # 如果没有短寿命核素，直接泰勒展开
                if not np.any(short_mask):
                    current_N = self._taylor_expansion(A, current_N, dt_sub)
                else:
                    # --- 阶段一: 长寿命核素求解 ---
                    
                    A_red = A.copy().tolil()
                    short_indices = np.where(short_mask)[0]
                    
                    # 移除短寿命核素的影响
                    for idx in short_indices:
                        A_red[idx, :] = 0.0 
                        A_red[:, idx] = 0.0 
                        
                    A_red = A_red.tocsr()
                    
                    # 更新长寿命核素的 N0
                    N_long_initial = current_N.copy()
                    N_long_initial[short_mask] = 0 
                    
                    N_next = self._taylor_expansion(A_red, N_long_initial, dt_sub)
                    
                    # --- 阶段二: 短寿命核素求解 (Secular Equilibrium) ---
    
                    with np.errstate(divide='ignore'):
                        inv_diag = 1.0 / removal_rates
                        inv_diag[long_mask] = 0.0 
                        
                    # 提取非对角矩阵 (源项矩阵)
                    A_off = A.copy()
                    A_off.setdiag(0)
                    
                    # 迭代求解
                    for k in range(self.origen_iter_limit):
                        # 计算总源项: S = A_off * N
                        source_term = A_off.dot(N_next)
                        
                        # 更新短寿命核素: N = Source / Removal
                        N_short_new = source_term * inv_diag
                        
                        # 将计算出的短寿命浓度填回总向量
                        N_next[short_mask] = N_short_new[short_mask]
                    
                    current_N = N_next
            elif self.method == 'cram':
                H = (A * dt).astype(np.complex128)
        
                # 准备单位矩阵 I (保持稀疏性)
                size = H.shape[0]
                I = eye(size, format='csr', dtype=np.complex128)
        
                # 2. 初始化结果 (极限项 alpha0 * N0)
                Nt = (cram_alpha0 * N0).astype(np.complex128)
        
                # 3. 循环计算部分分式求和
                # 公式: Nt = alpha0*N0 + 2 * Re( sum( alpha_j * (H - theta_j*I)^-1 * N0 ) )
                for k in range(len(cram_theta)):
                    theta_j = cram_theta[k]
                    alpha_j = cram_alpha[k]
            
                    # 将矩阵指数转化为线性方程的有理逼近
                    LHS = H - theta_j * I
            
                    # 构造右端项: RHS = alpha_j * N0
                    RHS = alpha_j * N0
            
                    # 求解线性方程组 LHS * x = RHS
                    x = spsolve(LHS, RHS)
            
                    # 累加结果 (利用共轭对称性，乘以 2 )
                    Nt += 2.0 * x
            
                # CRAM 计算结果理论上是实数，但会有微小的虚部计算噪声，所以取实部
                Nt = np.real(Nt)
        
                # 物理截断：浓度不能为负 (处理数值震荡)
                Nt[Nt < 0] = 0.0
                current_N = Nt
            
            current_N[current_N < 0] = 0.0
            
        return current_N

    def _taylor_expansion(self, A, N0, dt):
        """泰勒级数展开: exp(At) * N0"""
        N_curr = N0.copy()
        term = N0.copy() 
        
        for k in range(1, self.origen_order + 1):
            term = A.dot(term) * (dt / k)
            N_curr += term
            
            if np.max(np.abs(term)) < 1e-18 * np.max(np.abs(N_curr)):
                break
                
        return N_curr


# 每次裂变的平均能量 200 MeV
E_PER_FISSION = 3.204e-11 

def get_macroscopic_fission_cross_section(library, N_vector):
    """
    计算宏观裂变截面 Sigma_f = Sum(N_i * sigma_f_i)
    返回单位: cm^-1
    """
    Sigma_f = 0.0
    unit_conv = 1e-24 # Barn -> cm^2
    processed_isotopes = set()
    
    for p_idx, _, params in library.fission_events:
        if p_idx in processed_isotopes:
            continue
            
        sigma_f = params[0] # 获取裂变截面
        if sigma_f > 0:
            Sigma_f += N_vector[p_idx] * sigma_f * unit_conv
            processed_isotopes.add(p_idx)
            
    return Sigma_f

def run_simulation(mode, target_value, days, steps, N_initial, factory, solver, library):
    """
    Args:
        mode: 'constant_flux' 或 'constant_power'
        target_value: 
             如果是 constant_flux，值为通量 (n/cm^2/s)
             如果是 constant_power，值为功率密度 (W/cm^3)
        days: 总天数
        steps: 外步数
    """
    dt = (days * 86400) / steps
    time_points = [0]
    results = [N_initial.copy()]
    current_N = N_initial.copy()
    
    # 用于记录通量变化
    flux_history = [] 
    
    print(f"开始模拟: 模式={mode}, 目标值={target_value:.2e}")

    for step in range(steps):
        # --- 1. 定通量模式 (简单) ---
        if mode == 'constant_flux':
            phi = target_value
            flux_history.append(phi)
            
            # 直接求解
            A = factory.build_matrix(phi)
            current_N = solver.solve(A, current_N, dt)

        # --- 2. 定功率模式 (预测-校正) ---
        elif mode == 'constant_power':
            P_target = target_value
            
            # A. 初始状态 (Start of Step)
            Sigma_f_start = get_macroscopic_fission_cross_section(library, current_N)
            if Sigma_f_start == 0: raise ValueError("燃料耗尽，裂变截面为0")
            phi_start = P_target / (Sigma_f_start * E_PER_FISSION)
            A_start = factory.build_matrix(phi_start)
            N_pred = solver.solve(A_start, current_N, dt)
            Sigma_f_end = get_macroscopic_fission_cross_section(library, N_pred)
            phi_end = P_target / (Sigma_f_end * E_PER_FISSION)
            phi_avg = (phi_start + phi_end) / 2.0
            flux_history.append(phi_avg)
            
            A_avg = factory.build_matrix(phi_avg)
            current_N = solver.solve(A_avg, current_N, dt)

        results.append(current_N.copy())
        time_points.append((step + 1) * dt / 86400.0) 

    return np.array(results), time_points, flux_history

class SimulationEngine:
    def __init__(self, library_source, solver_method='scipy_expm', steps=10):  
        self.library = NuclearLibrary(json_source=library_source)
        self.factory = BurnupMatrixFactory(self.library)
        self.solver = SimpleSolver(method=solver_method)
        self.inner_steps = steps
        self.size = self.library.get_matrix_size()
        self.nuclides = self.library.idx_to_name
        self.idx_map = self.library.name_to_idx

    
# ==========================================
# 2. 前端 UI (Streamlit)
# ==========================================

# 页面配置
st.set_page_config(page_title="核燃料燃耗模拟器", page_icon="☢️", layout="wide")


st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white;}
    .reportview-container { background: #f0f2f6 }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：参数配置 ---
with st.sidebar:
    st.title("⚙️ 参数配置")
    
    st.subheader("0. 数据加载")
    uploaded_file = st.file_uploader("上传核数据库 (JSON)", type=["json"])
    
    st.subheader("1. 燃料参数")
    enrichment = st.slider("U-235 丰度 (%)", 0.7, 20.0, 3.0, 0.1)
    initial_density = st.number_input("初始铀原子数密度 (atoms/cm³)", value=2.4e22, format="%.2e")
    
    st.subheader("2. 运行模式")
    mode = st.radio("燃耗模式", ["定通量 (Constant Flux)", "定功率 (Constant Power)"])
    
    if mode == "定通量 (Constant Flux)":
        flux_input = st.number_input("热中子通量 (n/cm²/s)", value=3.0e13, format="%.2e")
        power_density = None
    else:
        power_density = st.number_input("功率密度 (W/cm³)", value=35.0, step=10.0)
        flux_input = None

    st.subheader("3. 时间设置")
    total_days = st.number_input("运行天数", value=300)
    steps = st.slider("计算步数 (Steps)", 10, 200, 50)
    inner_steps = st.number_input("每步内部分割数 (Inner Steps)", value=1, min_value=1, max_value=100)
    
    st.subheader("4. 求解器设置")
    solver_option = st.selectbox(
        "数值积分方法", 
        ["Matrix Exponential (Scipy)", "ORIGEN Method (泰勒展开 + 平衡近似)","CRAM Method (切比雪夫有理逼近法)"]
    )
    # 映射选项到内部方法名
    if "Matrix Exponential" in solver_option:
        solver_method = 'scipy_expm'
    elif "CRAM Method" in solver_option:
        solver_method = 'cram'
    else:
        solver_method = 'origen'

    st.subheader("5. 停堆设置")
    enable_shutdown = st.checkbox("模拟停堆 (Shutdown)")
    if enable_shutdown:
        shutdown_days = st.slider("停堆时长 (天)", 1, 10, 2)
        shutdown_steps = st.slider("停堆计算步数", 10, 100, 40)

    run_btn = st.button("🚀 开始计算")

# --- 主界面 ---
st.title("☢️ 核燃料燃耗演变模拟器")
st.markdown("---")

if run_btn:
    if uploaded_file is None:
        st.error("请先上传核数据库文件 (JSON)！")
        st.stop()

    # --- 初始化计算 ---
    try:
        engine = SimulationEngine(library_source=uploaded_file, solver_method=solver_method, steps=inner_steps)
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.stop()
        
    N = np.zeros(engine.size)
    
    # 初始装料 
    idx_u5 = engine.idx_map["U235"]
    idx_u8 = engine.idx_map["U238"]
    total_density = initial_density 
    N[idx_u5] = (enrichment / 100.0) * total_density
    N[idx_u8] = (1 - enrichment / 100.0) * total_density
    
    # 结果容器
    time_list = [0]
    results = [N.copy()]
    flux_log = [flux_input if flux_input else 0] # 记录通量变化
    
    dt = (total_days * 86400) / steps
    current_N = N.copy()
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- 阶段一：运行期间 ---
    phi = 0
    for s in range(steps):
        status_text.text(f"正在计算运行阶段: 第 {s+1}/{steps} 步")
        
        if mode == "定通量 (Constant Flux)":
            phi = flux_input
            A = engine.factory.build_matrix(phi,engine.library)
            current_N = engine.solver.solve(A, current_N, dt, steps=engine.inner_steps)
            flux_log.append(phi)
            
        else: # 定功率 
            # 1. 计算当前宏观裂变截面
            Sig_f = 0
            for p_idx, c_idx, params in engine.library.fission_events:
                sigma_f, y_val = params
                Sig_f += current_N[p_idx] * sigma_f * 1e-24
        
            # 2. 反推通量 (P = Phi * Sig_f * E)
            E_fiss = 3.2e-11
            phi = power_density / (Sig_f * E_fiss) if Sig_f > 0 else 0
            
            # 3. 求解
            A = engine.factory.build_matrix(phi, engine.library)
            current_N = engine.solver.solve(A, current_N, dt, steps=engine.inner_steps)
            flux_log.append(phi)
            
        current_N[current_N < 0] = 0
        results.append(current_N.copy())
        time_list.append((s + 1) * dt / 86400)
        progress_bar.progress((s + 1) / (steps + (shutdown_steps if enable_shutdown else 0)))

    # --- 阶段二：停堆期间 ---
    if enable_shutdown:
        dt_sd = (shutdown_days * 86400) / shutdown_steps
        phi_sd = 0.0 # 停堆通量为0
        
        for s in range(shutdown_steps):
            status_text.text(f"正在计算停堆阶段: 第 {s+1}/{shutdown_steps} 步")
            A = engine.factory.build_matrix(0, engine.library) # 只有衰变
            current_N = engine.solver.solve(A, current_N, dt_sd, steps=engine.inner_steps)
            current_N[current_N < 0] = 0
            
            results.append(current_N.copy())
            time_list.append(time_list[-1] + dt_sd / 86400)
            flux_log.append(0)
            progress_bar.progress((steps + s + 1) / (steps + shutdown_steps))

    progress_bar.empty()
    status_text.success("✅ 计算完成!")

    # --- 数据处理 ---
    res_arr = np.array(results)
    df = pd.DataFrame(res_arr, columns=engine.nuclides)
    df['Time (Days)'] = time_list
    df['Flux'] = flux_log
    
    # --- 结果展示 ---
    
    # 1. 关键指标卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最终燃耗天数", f"{time_list[-1]:.1f} d")
    c2.metric("U-235 剩余比例", f"{res_arr[-1, idx_u5]/res_arr[0, idx_u5]*100:.2f} %")
    c3.metric("Pu-239 积累密度", f"{res_arr[-1, engine.idx_map['Pu239']]:.2e}")
    c4.metric("最终通量水平", f"{flux_log[-2]:.2e}")

    # 2. 图表区域
    tab1, tab2, tab3 = st.tabs(["🔥 燃料与增殖", "☠️ 裂变毒物 (Xe/Sm)", "🌊 中子通量历史"])
    
    with tab1:
        st.subheader("主要锕系核素演变")
        # 选择要展示的核素
        actinides = ["U235", "U238", "Pu239", "Pu240", "Pu241"]
        fig1 = px.line(df, x='Time (Days)', y=actinides, log_y=True, 
                       labels={'value': 'Atom Density (atoms/cm³)'}, 
                       title="Actinide Inventory Evolution")
        # 添加停堆竖线
        if enable_shutdown:
            fig1.add_vline(x=total_days, line_dash="dash", line_color="green", annotation_text="Shutdown")
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        st.subheader("反应堆毒物演变")
        poisons = ["I135", "Xe135", "Pm149", "Sm149"]
        fig2 = px.line(df, x='Time (Days)', y=poisons, 
                       labels={'value': 'Atom Density (atoms/cm³)'},
                       title="Fission Product Poisons")
        if enable_shutdown:
            fig2.add_vline(x=total_days, line_dash="dash", line_color="green", annotation_text="Shutdown")
            
            # 碘坑区域
            st.info("💡 观察：在停堆线（绿色虚线）之后，Xe-135 的浓度先上升后下降,这是'碘坑'效应。")
            
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        st.subheader("中子通量随时间的变化")
        fig3 = px.line(df, x='Time (Days)', y='Flux',
                       title="Neutron Flux History")
        st.plotly_chart(fig3, use_container_width=True)
        if mode == "定功率 (Constant Power)":
            st.caption("注意：在定功率模式下，随着燃料消耗，通量必须上升以维持功率恒定。")

    # 3. 数据下载
    st.markdown("### 📥 数据导出")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("下载计算结果 (CSV)", csv, "burnup_results.csv", "text/csv")

else:

    st.info("👈 请在左侧调整参数，然后点击“开始计算”")

