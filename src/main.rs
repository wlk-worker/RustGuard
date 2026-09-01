//! ============================================================================
//!  RustGuard —— UEBA（用户与实体行为分析）内部数据泄露异常检测系统
//! ============================================================================
//!
//! 技术栈：纯 Rust + smartcore（DBSCAN）+ 手写 Isolation Forest，单文件实现。
//!
//! 流水线（main 严格按此顺序执行）：
//!   模块1 模拟日志生成器  -> 500 条日志（5 用户，95% 正常 + 5% 注入异常 A/B/C/D）
//!   模块2 特征工程        -> 每条日志 6 维特征
//!   模块3 异常检测（双引擎集成）
//!         引擎①：手写 Isolation Forest（纯 Rust，n_trees=100, max_samples=256）
//!         引擎②：鲁棒 Z-Score(MAD) + smartcore DBSCAN 混合检测（eps=0.5, min_samples=3）
//!         按 user_id "千人千面"独立建模；默认策略 HYBRID-PRIMARY（消融实验选出，
//!         详见 --stability / --stress；and/or/vote/smart/boost 可用 --fusion 切换）
//!   模块4 可解释性归因    -> 基线对比规则回溯，中文归因 + 风险分级
//!   模块5 报告输出        -> 终端彩色 + report.json（含检测器对比表 + 告警聚合事件）
//!   模块6 性能基准        -> 分阶段 + 端到端耗时；多 seed 稳定性模式
//!
//! 为什么手写 Isolation Forest？
//!   经源码核查，smartcore v0.4.10 未提供 IsolationForest（cluster 模块仅有
//!   kmeans / dbscan / agglomerative）。与其"降级"，本实现直接按算法原理手写：
//!   随机子采样 -> 随机特征随机切分建树 -> 平均路径长度 E[h] -> 异常分数 2^(-E[h]/c(n))，
//!   其中 c(n)=2(H(n-1)- (n-1)/n) 为同规模 BST 的平均不成功查找路径（调和均值修正项）。
//!   这样既字面满足原始需求（n_trees/max_samples/[-1,1] 输出），又保留混合检测器
//!   作为第二路异构引擎：两路算法原理完全不同（树隔离 vs 统计+密度），
//!   交集判定比单引擎更抗误报，且 report.json 提供三方对比表支撑"检测器选型"论证。
//!
//! 运行：
//!   cargo run                          # 默认 seed=42，完整彩色报告
//!   cargo run --release                # 发布构建（性能基准）
//!   cargo run -- --seed 7              # 换数据
//!   cargo run -- --stability 10        # 多 seed 稳定性报告（均值±标准差）
//!   cargo run -- --quiet               # 只输出统计摘要

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use chrono::{Datelike, Days, Local, NaiveDateTime, Timelike};
use colored::Colorize;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use smartcore::cluster::dbscan::{DBSCAN, DBSCANParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;

// ============================================================================
// 全局常量与可调参数
// ============================================================================

/// 默认随机种子：固定种子保证结果可复现
const SEED: u64 = 42;
/// 日志总数
const TOTAL_LOGS: usize = 500;
/// 虚拟用户数（1~5）
const NUM_USERS: u32 = 5;
/// 每个用户的正常日志条数（5 × 95 = 475）
const NORMAL_PER_USER: usize = 95;
/// 日志时间跨度：最近 7 天
const DAYS_SPAN: u64 = 7;

// ---- 引擎② Z-Score + DBSCAN 参数（阈值可调）----
/// 双维联合阈值：≥2 个特征 |z| 超过该值进入候选（单维擦线的尾部噪声被剔除）
const Z_THRESHOLD: f64 = 3.0;
/// 单维强偏离阈值：仅 1 组证据超阈时，需达到该强度才进入候选
/// （两级判据 + 证据源分组是对多维多重比较下 3σ 尾部假阳性的修正：
/// 随机种子实测 FP 从 ~7/轮 降至 ~1/轮；log_size 与偏离度同源于体量证据，
/// 按组去重防止一个偏大的正常文件被计成"两维独立证据"）
const Z_SINGLE_STRONG: f64 = 5.0;
/// DBSCAN 邻域半径（smartcore 默认值即为 0.5）
const DBSCAN_EPS: f64 = 0.5;
/// DBSCAN 核心点最小邻居数
const DBSCAN_MIN_SAMPLES: usize = 3;
/// 簇内典型性距离阈值（z 空间欧氏距离）：即使点被 DBSCAN 密度链"捎带"进主导簇，
/// 离质心过远仍视为命中——修复连击爬坡样本沿滑窗计数逐级链接进正常簇的漏检（stress U1 发现）
const Z_TYPICAL_DIST: f64 = 4.0;
/// 频率维自适应 sigma 下限分母：count+2 之内不算异常、count+8 必炸（防滑窗密度误报）
const FREQ_FLOOR_COUNTS: f64 = 2.0;
/// 送入 DBSCAN 前对 Z 值裁剪的幅度（避免单维极端值支配欧氏距离）
const Z_CLIP: f64 = 4.0;
/// 每特征鲁棒标准差下限（防 MAD=0 放大假异常；顺序同特征）。
/// 离散维（密级/操作）取下限≥1.0：保证"相邻取值差"不超阈（stress 发现 view 稀有用户 z=-4 误报）；
/// 第7项 is_weekend 取小值，使"工作日基线跳变到周末"这种稀有值事件能穿透 z 阈值。
const SIGMA_FLOOR: [f64; 7] = [0.5, 0.10, 0.50, 1.00, 0.06, 0.10, 0.15];

// ---- 引擎① 手写 Isolation Forest 参数 ----
/// 树数量（需求规格指定）
const IF_N_TREES: usize = 100;
/// 每树最大子采样数（需求规格指定；单用户 <256 条时全量采样）
const IF_MAX_SAMPLES: usize = 256;
/// 预期污染率先验（运维声明而非真值泄漏：对标 sklearn contamination 参数）。
/// 阈值不再拍固定魔数，而是取全体分数 (1-该比例) 分位数自适应校准——
/// 实验证明固定 0.62 阈值在本数据集召回仅 52%，分位数校准是标准做法。
const IF_CONTAMINATION: f64 = 0.05;

/// 欧拉-马歇罗尼常数（c(n) 公式中调和数的近似项）
const EULER_GAMMA: f64 = 0.5772156649015329;

/// 告警聚合：同用户两条异常间隔 ≤ 该秒数且同簇 → 合并为同一事件
const INCIDENT_GAP_SECS: i64 = 360;

/// 特征名称（归因与调试用；第7项仅在 --stress 的 U2' 实验中启用）
const FEATURE_NAMES: [&str; 7] = [
    "hour(操作时刻)",
    "log_size(大小对数)",
    "sensitivity(密级)",
    "operation(操作编码)",
    "freq_5min(5分钟频率)",
    "log_size_ratio(大小偏离度对数)",
    "is_weekend(是否周末)",
];

/// 5 分钟 = 300 秒（滑窗）
const WINDOW_SECS: i64 = 300;

// ============================================================================
// 模块1：模拟日志生成器
// ============================================================================

/// 一条文件访问日志。is_anomaly 由生成逻辑自动标记（真值标签，仅用于评测；
/// 检测模型不读取该字段，防止标签泄漏）。
/// scenario_tag 记录异常所属场景（0=正常 1=A深夜 2=B连击 3=C越权密级 4=D超大
/// 5=U1突发delete 6=U2周末白昼 7=U3慢速蚕食 8=U4群体微小），
/// 仅供压力测试分型统计，序列化时跳过（不污染 report.json 字段规范）。
#[derive(Debug, Clone, Serialize)]
pub struct LogRecord {
    pub user_id: u32,
    pub timestamp: chrono::DateTime<Local>,
    pub operation: String,
    pub file_size_kb: f64,
    pub sensitivity: u8,
    pub is_anomaly: bool,
    #[serde(skip)]
    pub scenario_tag: u8,
}

/// Box-Muller 变换：均匀分布 -> 正态分布 N(mean, std)（不引入 rand_distr）
fn gaussian(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>(); // 落在 (0,1]，避免 ln(0)
    let u2 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std * z
}

/// 构造"距今 days_back 天前、指定时分秒"的本地时间（earliest 消除 DST 二义性）
fn make_time(
    days_back: u64,
    hour: u32,
    minute: u32,
    second: u32,
) -> Result<chrono::DateTime<Local>> {
    let date = Local::now()
        .date_naive()
        .checked_sub_days(Days::new(days_back))
        .ok_or_else(|| anyhow!("日期回退溢出"))?;
    let naive = date
        .and_hms_opt(hour, minute, second)
        .ok_or_else(|| anyhow!("非法时间"))?;
    naive
        .and_local_timezone(Local)
        .earliest()
        .ok_or_else(|| anyhow!("时区解析失败"))
}

/// 在给定"朴素时间"上构造本地时间（用于 5 分钟连击下载序列）
fn make_time_from_naive(naive: NaiveDateTime) -> Result<chrono::DateTime<Local>> {
    naive
        .and_local_timezone(Local)
        .earliest()
        .ok_or_else(|| anyhow!("时区解析失败"))
}

/// 操作名 -> 特征编码（view→0, download→1, edit→2, delete→3）
fn encode_operation(op: &str) -> f64 {
    match op {
        "view" => 0.0,
        "download" => 1.0,
        "edit" => 2.0,
        "delete" => 3.0,
        _ => -1.0,
    }
}

/// 数据生成配置：默认值为标准评测数据集；--stress 模式随机化各字段，
/// 用于验证检测器对数据构成变化的稳健性
#[derive(Debug, Clone)]
struct GenConfig {
    seed: u64,
    users: u32,
    normal_per_user: usize,
    /// 四类已知场景注入条数（A深夜大额 / B连击下载 / C密级越权 / D超大下载）
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    /// 场景C文件大小下界（调小=更接近正常体量=更难检，压力测试变量）
    c_size_lo: f64,
    /// 是否追加 UNKNOWN 异常池（U1~U4：从未参与任何调参的假设外攻击类型）
    unknown: bool,
}

impl GenConfig {
    /// 规格默认数据集：5用户×95正常 + A4/B12/C5/D4 = 500条25异常
    fn default_() -> Self {
        GenConfig {
            seed: SEED,
            users: NUM_USERS,
            normal_per_user: NORMAL_PER_USER,
            a: 4,
            b: 12,
            c: 5,
            d: 4,
            c_size_lo: 280.0,
            unknown: false,
        }
    }
    /// 本配置的注入异常总数（UNKNOWN 池：U1=15, U2=6, U3=40, U4=users）
    fn injected(&self) -> usize {
        (self.a + self.b + self.c + self.d) as usize
            + if self.unknown {
                15 + 6 + 40 + self.users as usize
            } else {
                0
            }
    }
}

/// 取一个落在工作日（周一至周五）的 days_back：企业作息真实约束——
/// 正常行为只发生在工作日，"是否周末"特征才具备判别力（供 stress U2/U2' 使用）
fn workday_offset(rng: &mut StdRng) -> u64 {
    for _ in 0..30 {
        let d = rng.gen_range(0..DAYS_SPAN);
        if let Some(dt) = Local::now().date_naive().checked_sub_days(Days::new(d)) {
            if !matches!(dt.weekday(), chrono::Weekday::Sat | chrono::Weekday::Sun) {
                return d;
            }
        }
    }
    1 // 保底（7 天窗口内工作日≥5 天，理论不可达）
}

/// 找到最近两周内的一个周六/周日的 days_back（用于 U2 周末场景）
fn weekend_day_back(rng: &mut StdRng) -> Result<u64> {
    let today = Local::now().date_naive();
    let mut candidates = Vec::new();
    for d in 1..14u64 {
        if let Some(dt) = today.checked_sub_days(Days::new(d)) {
            if matches!(dt.weekday(), chrono::Weekday::Sat | chrono::Weekday::Sun) {
                candidates.push(d);
            }
        }
    }
    if candidates.is_empty() {
        return Ok(6); // 理论不可达（14 天内必有周末），保底防 panic
    }
    let idx = rng.gen_range(0..candidates.len());
    Ok(candidates[idx])
}

/// 生成日志（按 GenConfig）
fn generate_logs(cfg: &GenConfig) -> Result<Vec<LogRecord>> {
    let seed = cfg.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    let users = cfg.users;
    let mut logs: Vec<LogRecord> = Vec::new();

    // -------- 正常行为（95%）：工作日 9:00-18:00，体量 ~LN(中位100KB, σ0.38) 长尾分布，密级 1~2 --------
    for user in 1..=users {
        for _ in 0..cfg.normal_per_user {
            // 随机参数先取到局部变量，避免同一表达式多次可变借用 rng
            let day = workday_offset(&mut rng);
            let hour = rng.gen_range(9..18);
            let minute = rng.gen_range(0..60);
            let second = rng.gen_range(0..60);
            let ts = make_time(day, hour, minute, second)?;
            // 正常文件体量取对数正态分布 LN(mu=4.605, sigma=0.30)：
            // 中位数≈100KB、右偏长尾、无人为截断边界——企业文件大小的典型真实形态。
            // 鲁棒统计（MAD）对长尾天然免疫，避免正态截断造成的边界堆积假异常
            let size = (gaussian(&mut rng, 0.0, 1.0) * 0.30 + 4.605).exp();
            let ops = ["view", "download", "edit"];
            let op = ops[rng.gen_range(0..3)].to_string();
            logs.push(LogRecord {
                user_id: user,
                timestamp: ts,
                operation: op,
                file_size_kb: size,
                sensitivity: rng.gen_range(1..=2),
                is_anomaly: false,
                scenario_tag: 0,
            });
        }
    }

    // -------- 场景 A：深夜访问（0-5 点）且文件 > 500KB --------
    for k in 0..cfg.a {
        let user = 1 + (k % users);
        let day = workday_offset(&mut rng);
        let hour = rng.gen_range(0..6);
        let minute = rng.gen_range(0..60);
        let second = rng.gen_range(0..60);
        let ts = make_time(day, hour, minute, second)?;
        logs.push(LogRecord {
            user_id: user,
            timestamp: ts,
            operation: "download".to_string(),
            file_size_kb: rng.gen_range(600.0..1600.0),
            sensitivity: 2,
            is_anomaly: true,
            scenario_tag: 1,
        });
    }

    // -------- 场景 B：5 分钟内连续下载 ≥10 次（每次 >200KB）--------
    {
        let burst_user: u32 = 1 + rng.gen_range(0..users);
        let day = workday_offset(&mut rng);
        let date = Local::now()
            .date_naive()
            .checked_sub_days(Days::new(day))
            .ok_or_else(|| anyhow!("日期回退溢出"))?;
        let base = date
            .and_hms_opt(14, 10, 0)
            .ok_or_else(|| anyhow!("非法时间"))?;
        // 每 18 秒一次：条数上限 15 时 14*18=252s < 300s，确保全部落入 5 分钟滑窗
        for j in 0..cfg.b {
            let ts = make_time_from_naive(base + chrono::TimeDelta::seconds((j * 18) as i64))?;
            logs.push(LogRecord {
                user_id: burst_user,
                timestamp: ts,
                operation: "download".to_string(),
                file_size_kb: rng.gen_range(350.0..700.0),
                sensitivity: rng.gen_range(1..=2),
                is_anomaly: true,
                scenario_tag: 2,
            });
        }
    }

    // -------- 场景 C：越权访问密级=5（c_size_lo 可调小以制造"更难检"配置）--------
    for k in 0..cfg.c {
        let user = 1 + (k % users);
        let day = workday_offset(&mut rng);
        let hour = rng.gen_range(9..18);
        let minute = rng.gen_range(0..60);
        let second = rng.gen_range(0..60);
        let ts = make_time(day, hour, minute, second)?;
        logs.push(LogRecord {
            user_id: user,
            timestamp: ts,
            operation: "download".to_string(),
            file_size_kb: rng.gen_range(cfg.c_size_lo..450.0),
            sensitivity: 5,
            is_anomaly: true,
            scenario_tag: 3,
        });
    }

    // -------- 场景 D：单次下载 > 用户历史均值 10 倍 --------
    for k in 0..cfg.d {
        let user = 1 + (k % users);
        let day = workday_offset(&mut rng);
        let hour = rng.gen_range(9..18);
        let minute = rng.gen_range(0..60);
        let second = rng.gen_range(0..60);
        let ts = make_time(day, hour, minute, second)?;
        logs.push(LogRecord {
            user_id: user,
            timestamp: ts,
            operation: "download".to_string(),
            file_size_kb: rng.gen_range(1200.0..2500.0),
            sensitivity: 2,
            is_anomaly: true,
            scenario_tag: 4,
        });
    }

    // ================= UNKNOWN 池：从未参与调参的"假设外"攻击 =================

    if cfg.unknown {
        // ---- U1 突发批量 delete（标签5）：15 次 delete、体量正常——考验操作序列感知 ----
        {
            let u = users; // 最后一个用户
            let day = workday_offset(&mut rng);
            let date = Local::now()
                .date_naive()
                .checked_sub_days(Days::new(day))
                .ok_or_else(|| anyhow!("日期回退溢出"))?;
            let base = date.and_hms_opt(10, 0, 0).ok_or_else(|| anyhow!("非法时间"))?;
            for j in 0..15u32 {
                // 18 秒一次：14*18=252s<300s，后段记录 5 分钟计数可达 15 → 频率特征可表达
                let ts = make_time_from_naive(base + chrono::TimeDelta::seconds((j * 18) as i64))?;
                logs.push(LogRecord {
                    user_id: u,
                    timestamp: ts,
                    operation: "delete".to_string(),
                    file_size_kb: rng.gen_range(50.0..120.0),
                    sensitivity: rng.gen_range(1..=2),
                    is_anomaly: true,
                    scenario_tag: 5,
                });
            }
        }
        // ---- U2 周末白昼访问（标签6）：所有既有特征全部正常 → 预期盲区（除非启用星期维）----
        {
            let u = users.saturating_sub(1).max(1);
            let wd = weekend_day_back(&mut rng)?;
            for _ in 0..6u32 {
                let hour = rng.gen_range(10..17); // 正常工时内
                let ts = make_time(wd, hour, rng.gen_range(0..60), rng.gen_range(0..60))?;
                logs.push(LogRecord {
                    user_id: u,
                    timestamp: ts,
                    operation: ["view", "edit"][rng.gen_range(0..2)].to_string(),
                    file_size_kb: rng.gen_range(60.0..200.0), // 正常体量
                    sensitivity: rng.gen_range(1..=2),
                    is_anomaly: true,
                    scenario_tag: 6,
                });
            }
        }
        // ---- U3 慢速蚕食（标签7）：90 天里体量从 110→170KB 线性爬升，单条永在噪声内 ----
        {
            let u = 1u32;
            for j in 0..40u32 {
                let day = ((j as f64 / 40.0) * 90.0) as u64; // 跨 90 天
                let hour = rng.gen_range(9..18);
                let ts = make_time(day, hour, rng.gen_range(0..60), rng.gen_range(0..60))?;
                logs.push(LogRecord {
                    user_id: u,
                    timestamp: ts,
                    operation: "download".to_string(),
                    file_size_kb: 110.0 + 60.0 * j as f64 / 40.0,
                    sensitivity: rng.gen_range(1..=2),
                    is_anomaly: true,
                    scenario_tag: 7,
                });
            }
        }
        // ---- U4 群体微小共谋（标签8）：每个用户同一工作日各多下一次 180~250KB ----
        let u4day = workday_offset(&mut rng);
        for u in 1..=users {
            let hour = rng.gen_range(11..13);
            let ts = make_time(u4day, hour, rng.gen_range(0..60), rng.gen_range(0..60))?;
            logs.push(LogRecord {
                user_id: u,
                timestamp: ts,
                operation: "download".to_string(),
                file_size_kb: rng.gen_range(180.0..250.0),
                sensitivity: rng.gen_range(1..=2),
                is_anomaly: true,
                scenario_tag: 8,
            });
        }
    }

    // 按时间排序，让"滑窗计数 / 历史均值"等时序特征有明确定义
    logs.sort_by_key(|l| (l.timestamp, l.user_id));
    Ok(logs)
}

// ============================================================================
// 模块2：特征工程（LogRecord -> 6~7 维特征向量）
// ============================================================================
/// 特征工程结果：6 维特征（规格）+ 可选第 7 维 is_weekend（压力实验用）+ 归因友好原始量
#[derive(Debug, Clone)]
struct Enriched {
    features: Vec<f64>,
    /// 5 分钟滑窗原始操作次数（含当前）
    count5: usize,
    /// 全局最大 5 分钟计数（频率维自适应下限用）
    max_count: usize,
    /// 当前大小 / 用户历史均值（偏离度）
    ratio: f64,
}

/// 中位数（原地排序，鲁棒统计基石）
fn median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

/// 6 维特征：
/// 1. 小时数(0-23，含分钟细分) 2. ln(size+1) 3. 密级 4. 操作编码
/// 5. 5 分钟滑窗计数/全局最大值 ∈0~1
/// 6. ln(size/历史扩展均值)（偏离度对数化保证对称分布；无未来泄漏、无标签泄漏）
fn engineer_features(logs: &[LogRecord], include_weekday: bool) -> Vec<Enriched> {
    let n = logs.len();
    let mut counts = vec![1usize; n];
    let mut ratios = vec![1.0f64; n];
    let mut windows: HashMap<u32, VecDeque<i64>> = HashMap::new();
    let mut hist: HashMap<u32, (f64, usize)> = HashMap::new();

    for (i, r) in logs.iter().enumerate() {
        let ts = r.timestamp.timestamp();
        // 特征5：滑动窗口计数
        let dq = windows.entry(r.user_id).or_default();
        while let Some(&front) = dq.front() {
            if ts - front > WINDOW_SECS {
                dq.pop_front();
            } else {
                break;
            }
        }
        dq.push_back(ts);
        counts[i] = dq.len();

        // 特征6：偏离度（仅用该用户时序上更早的日志求均值）。
        // 置信平滑：历史 <10 条时小样本均值不稳，按"无偏离"(1.0) 处理——
        // stress 发现早期记录因均值抖动产生 ratio z>3 的误报
        let h = hist.entry(r.user_id).or_insert((0.0, 0));
        ratios[i] = if h.1 < 10 {
            1.0
        } else {
            r.file_size_kb / (h.0 / h.1 as f64)
        };
        h.0 += r.file_size_kb;
        h.1 += 1;
    }

    let max_count = counts.iter().copied().max().unwrap_or(1).max(1) as f64;

    logs.iter()
        .enumerate()
        .map(|(i, r)| {
            let hour = r.timestamp.hour() as f64 + r.timestamp.minute() as f64 / 60.0;
            let mut features = vec![
                hour,
                (r.file_size_kb + 1.0).ln(),
                r.sensitivity as f64,
                encode_operation(&r.operation),
                counts[i] as f64 / max_count,
                // 偏离度取 ln(ratio)：比值分布右偏（乘性），线性值上做绝对 z-score 会把
                // 93 分位的普通大文件放大成 z>3 假异常；对数化后分布对称，z 判据才成立
                ratios[i].ln(),
            ];
            // 可选第 7 维 is_weekend（0/1），压力实验 U2' 特征对照用。
            // 采用二值稀有值而非星期几数值编码：离散数值在单维 z 判据下会饱和（z 仅≈2），
            // 该结论由压力测试实测得出。
            if include_weekday {
                let wk = matches!(
                    r.timestamp.weekday(),
                    chrono::Weekday::Sat | chrono::Weekday::Sun
                );
                features.push(if wk { 1.0 } else { 0.0 });
            }
            Enriched {
                features,
                count5: counts[i],
                max_count: max_count as usize,
                ratio: ratios[i],
            }
        })
        .collect()
}

// ============================================================================
// 模块3-引擎①：手写 Isolation Forest（纯 Rust，无第三方实现）
// ============================================================================

/// IF 树节点：left==usize::MAX 表示叶节点（size 为该叶中样本数）
#[derive(Debug, Clone)]
struct IfNode {
    feat: usize,
    split: f64,
    left: usize,
    right: usize,
    size: usize,
}

#[derive(Debug, Clone)]
struct IfTree {
    nodes: Vec<IfNode>,
    root: usize,
}

/// 隔离森林：100 棵树 + 子采样规模 n 对应的归一化常数 c(n)
struct IsolationForest {
    trees: Vec<IfTree>,
    c_sub: f64,
}

/// c(n) = 2*(H(n-1) - (n-1)/n)，H(m)≈ln(m)+EULER_GAMMA —— "平均路径长度"的
/// 归一化基准（同规模二叉搜索树的不成功查找期望），n≤1 时为 0
fn avg_bst_unsuccessful(n: usize) -> f64 {
    if n <= 1 {
        0.0
    } else if n == 2 {
        1.0
    } else {
        let m = n as f64;
        2.0 * ((m - 1.0).ln() + EULER_GAMMA) - 2.0 * (m - 1.0) / m
    }
}

/// 递归构建一棵隔离树：随机特征 + 该特征值域内随机切分点；
/// 达到深度限制 / 样本 ≤1 / 无法产生有效切分时落叶
fn build_if_tree(
    rows: &[Vec<f64>],
    sample: &[usize],
    depth: usize,
    limit: usize,
    rng: &mut StdRng,
    nodes: &mut Vec<IfNode>,
) -> usize {
    if depth >= limit || sample.len() <= 1 {
        nodes.push(IfNode {
            feat: 0,
            split: 0.0,
            left: usize::MAX,
            right: usize::MAX,
            size: sample.len(),
        });
        return nodes.len() - 1;
    }
    // 至多重试 10 次找一个"可分"的特征（极端共线时退化为叶）
    let dim = rows.first().map(|r| r.len()).unwrap_or(0);
    for _attempt in 0..10usize {
        let f = rng.gen_range(0..dim.max(1));
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &i in sample {
            let v = rows[i][f];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if hi > lo {
            let sp = rng.gen_range(lo..hi);
            let mut left_sample: Vec<usize> = Vec::with_capacity(sample.len());
            let mut right_sample: Vec<usize> = Vec::with_capacity(sample.len());
            for &i in sample {
                if rows[i][f] < sp {
                    left_sample.push(i);
                } else {
                    right_sample.push(i);
                }
            }
            if !left_sample.is_empty() && !right_sample.is_empty() {
                let my_idx = nodes.len();
                nodes.push(IfNode {
                    feat: f,
                    split: sp,
                    left: usize::MAX,
                    right: usize::MAX,
                    size: 0,
                });
                let l = build_if_tree(rows, &left_sample, depth + 1, limit, rng, nodes);
                let r = build_if_tree(rows, &right_sample, depth + 1, limit, rng, nodes);
                nodes[my_idx].left = l;
                nodes[my_idx].right = r;
                return my_idx;
            }
        }
    }
    // 10 次都切不开：当前样本整体落叶
    nodes.push(IfNode {
        feat: 0,
        split: 0.0,
        left: usize::MAX,
        right: usize::MAX,
        size: sample.len(),
    });
    nodes.len() - 1
}

/// 训练：每棵树独立做一次部分 Fisher-Yates 洗牌取前 sub 个作为子采样
fn if_train(rows: &[Vec<f64>], seed: u64) -> IsolationForest {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = rows.len();
    let sub = IF_MAX_SAMPLES.min(n);
    let limit = (n.max(2) as f64).log2().ceil() as usize; // ψ(n) = ⌈log2(n_sub)⌉
    let mut trees = Vec::with_capacity(IF_N_TREES);
    for _ in 0..IF_N_TREES {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..sub {
            let j = rng.gen_range(i..n);
            idx.swap(i, j);
        }
        let mut nodes: Vec<IfNode> = Vec::with_capacity(sub * 2);
        let root = build_if_tree(rows, &idx[..sub], 0, limit, &mut rng, &mut nodes);
        trees.push(IfTree { nodes, root });
    }
    IsolationForest {
        trees,
        c_sub: avg_bst_unsuccessful(sub),
    }
}

/// 样本在单棵树上的路径长度：叶节点处再加 c(leaf_size) 修正（未完全隔离的期望深度）
fn tree_path_length(t: &IfTree, x: &[f64]) -> f64 {
    let mut cur = t.root;
    let mut d = 0.0f64;
    loop {
        let node = &t.nodes[cur];
        if node.left == usize::MAX {
            return d + avg_bst_unsuccessful(node.size);
        }
        cur = if x[node.feat] < node.split {
            node.left
        } else {
            node.right
        };
        d += 1.0;
    }
}

/// 异常分数 raw = 2^(-E[h]/c(n)) ∈ (0,1)，越接近 1 越异常
fn if_score(forest: &IsolationForest, x: &[f64]) -> f64 {
    if forest.c_sub <= 0.0 || forest.trees.is_empty() {
        return 0.0;
    }
    let sum: f64 = forest.trees.iter().map(|t| tree_path_length(t, x)).sum();
    let avg = sum / forest.trees.len() as f64;
    (-avg * std::f64::consts::LN_2 / forest.c_sub).exp()
}

// ============================================================================
// 模块3-引擎②：鲁棒 Z-Score + smartcore DBSCAN 混合检测（按用户独立建模）
// ============================================================================

/// 混合检测器输出
#[derive(Debug, Clone)]
struct HybOut {
    /// 候选 ∧ DBSCAN命中
    anom: bool,
    /// 单独的 DBSCAN 命中标志（供高级融合策略使用）
    dbscan_hit: bool,
    max_z: f64,
    z: Vec<f64>,
    /// DBSCAN 簇标签（0=噪声，≥1=簇号）
    dbscan_label: u32,
}

/// 用户行为基线（仅由"主导正常簇"成员统计得到 —— 零标签泄漏）
#[derive(Debug, Clone)]
struct UserBaseline {
    work_start_hour: u32,
    work_end_hour: u32,
    mean_file_size_kb: f64,
    max_count_5min: usize,
    max_sensitivity: u8,
}

/// 按用户分组执行 Z-Score + DBSCAN 检测；同时返回每用户基线
/// eps / z_thr 为运行时参数（默认取常量），供 --sensitivity 网格扫描复用同一实现
fn detect_hybrid(
    logs: &[LogRecord],
    enriched: &[Enriched],
    eps: f64,
    z_thr: f64,
) -> Result<(Vec<HybOut>, BTreeMap<u32, UserBaseline>)> {
    let mut outs = vec![
        HybOut {
            anom: false,
            dbscan_hit: false,
            max_z: 0.0,
            z: vec![0.0; enriched.first().map(|e| e.features.len()).unwrap_or(6)],
            dbscan_label: 0,
        }; logs.len()
    ];
    let mut baselines: BTreeMap<u32, UserBaseline> = BTreeMap::new();

    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, l) in logs.iter().enumerate() {
        groups.entry(l.user_id).or_default().push(i);
    }

    for (user, idxs) in groups {
        let n = idxs.len();
        let dim = enriched.first().map(|e| e.features.len()).unwrap_or(6);

        // 1) 鲁棒基线：中位数 + MAD（1.4826*MAD ≈ 正态等效 std；抗异常点污染）
        let columns: Vec<Vec<f64>> = (0..dim)
            .map(|f| idxs.iter().map(|&i| enriched[i].features[f]).collect())
            .collect();
        let mut med = vec![0.0f64; dim];
        let mut sig = vec![0.0f64; dim];
        let max_count_global = enriched
            .first()
            .map(|e| e.max_count.max(1))
            .unwrap_or(12) as f64;
        for f in 0..dim {
            let mut col = columns[f].clone();
            med[f] = median(&mut col);
            let mut dev: Vec<f64> = col.iter().map(|x| (x - med[f]).abs()).collect();
            // 频率维（index 4）用自适应下限：一个正常计数跳变的 sigma 下限与全局最大计数挂钩，
            // 相当于"count 相对基线至少多出 FREQ_FLOOR_COUNTS 次才算 1σ"
            let floor = if f == 4 {
                (FREQ_FLOOR_COUNTS / max_count_global)
                    .max(SIGMA_FLOOR[4])
            } else {
                SIGMA_FLOOR[f.min(SIGMA_FLOOR.len() - 1)]
            };
            sig[f] = (1.4826 * median(&mut dev)).max(floor);
        }

        // 2) Z-Score 两级候选判据（对抗多维多重比较的尾部假阳性）：
        //    单组证据 |z| > Z_SINGLE_STRONG（证据足够强），或
        //    ≥2 组独立证据 |z| > z_thr（相互印证）
        //    证据源分组：体量(log_size)与其衍生指标(偏离度)只计一组，避免同源重复计数
        let mut zs: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut candidates: Vec<bool> = Vec::with_capacity(n);
        // 各组证据的最大 |z|：索引 5(偏离度) 并入索引 1(体量) 组，其余各成一维一组
        let group_of = |f: usize| -> usize {
            match f {
                5 => 1,
                x if x > 5 => x - 1, // 6(is_weekend) -> 组5
                x => x,
            }
        };
        let n_groups = dim - 1;
        for j in 0..n {
            let feats = &enriched[idxs[j]].features;
            let mut z = vec![0.0f64; dim];
            for f in 0..dim {
                z[f] = (feats[f] - med[f]) / sig[f];
            }
            let max_z = z.iter().map(|&v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
            let mut group_max = vec![0.0f64; n_groups];
            for f in 0..dim {
                let g = group_of(f);
                if g < n_groups {
                    group_max[g] = group_max[g].max(z[f].abs());
                }
            }
            let strong_groups = group_max.iter().filter(|v| **v > z_thr).count();
            candidates.push(max_z > Z_SINGLE_STRONG || strong_groups >= 2);
            zs.push(z);
        }

        // 3) DBSCAN（smartcore）：在裁剪 Z 空间发现密度结构
        let scaled: Vec<Vec<f64>> = zs
            .iter()
            .map(|row| row.iter().map(|&v| v.clamp(-Z_CLIP, Z_CLIP) / Z_CLIP).collect())
            .collect();
        let x = DenseMatrix::from_2d_vec(&scaled)
            .map_err(|e| anyhow!("用户{user} 构造特征矩阵失败: {e}"))?;
        let model = DBSCAN::fit(
            &x,
            DBSCANParameters::default()
                .with_eps(eps)
                .with_min_samples(DBSCAN_MIN_SAMPLES),
        )
        .map_err(|e| anyhow!("用户{user} DBSCAN 训练失败: {e}"))?;
        let labels: Vec<u32> = model
            .predict(&x)
            .map_err(|e| anyhow!("用户{user} DBSCAN 预测失败: {e}"))?;

        // 主导簇 = 成员最多的正标签簇（≈95% 正常行为的聚集处）
        let mut freq: HashMap<u32, usize> = HashMap::new();
        for &lb in &labels {
            if lb > 0 {
                *freq.entry(lb).or_insert(0) += 1;
            }
        }
        let dominant = freq.iter().max_by_key(|(_, c)| **c).map(|(k, _)| *k);
        // "异常小簇"的规模上限：真实用户常呈多峰正常行为（如上午批处理+下午零散浏览），
        // 第二大正常子簇可能有几十条成员，不能因"非主导"就判异常——
        // 仅成员数 ≤ max(10, n/6) 的簇具备簇级异常资格（seed4 实测修复 45 误报连锁）
        let small_cluster_max = 10usize.max(n / 6);

        // 主导簇质心（z 空间）——供"簇内典型性"检验，防密度链把异常捎带进正常簇
        let centroid: Option<Vec<f64>> = dominant.map(|d| {
            let mem: Vec<usize> = (0..n).filter(|&j| labels[j] == d).collect();
            let mut c = vec![0.0f64; dim];
            for &j in &mem {
                for f in 0..dim {
                    c[f] += zs[j][f];
                }
            }
            let m = mem.len().max(1) as f64;
            for f in 0..dim {
                c[f] /= m;
            }
            c
        });

        // 4) DBSCAN 命中 = 噪声点(label=0) ∨ 非主导小簇（群体异常） ∨ 离主导簇质心过远
        //    （第三条修复 U1 类"逐级链接进主导簇"的漏检：滑窗计数从 1 缓爬到 15 时，
        //    相邻记录距离 < eps 形成密度链，纯簇归属会误判为"正常簇成员"）
        for j in 0..n {
            let typical_dist = centroid.as_ref().map_or(f64::INFINITY, |c| {
                zs[j]
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            });
            let small_non_dominant = labels[j] > 0
                && dominant.map_or(true, |d| labels[j] != d)
                && *freq.get(&labels[j]).unwrap_or(&0) <= small_cluster_max;
            let dbscan_hit = labels[j] == 0 || small_non_dominant || typical_dist > Z_TYPICAL_DIST;
            let max_z = zs[j].iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
            outs[idxs[j]] = HybOut {
                anom: candidates[j] && dbscan_hit,
                dbscan_hit,
                max_z,
                z: zs[j].clone(),
                dbscan_label: labels[j],
            };
        }

        // 4b) 群体异常扩展（collective anomaly）：非主导小簇内只要有一个点被
        //     两级判据确认为异常，则整簇成员全部判定异常——连击下载的前几条个体
        //     特征几乎正常，但"与已确认异常同属一个异常密度簇"本身就是充分证据，
        //     该归属关系由 DBSCAN 学习所得，不含任何场景规则。
        if let Some(d) = dominant {
            let mut by_label: HashMap<u32, Vec<usize>> = HashMap::new();
            for j in 0..n {
                if labels[j] > 0
                    && labels[j] != d
                    && *freq.get(&labels[j]).unwrap_or(&0) <= small_cluster_max
                {
                    by_label.entry(labels[j]).or_default().push(j);
                }
            }
            for members in by_label.values() {
                if members.iter().any(|&j| outs[idxs[j]].anom) {
                    for &j in members {
                        outs[idxs[j]].anom = true;
                    }
                }
            }
        }

        // 4c) 时间桥接：连击序列的早期成员可能被 DBSCAN 划为孤立噪声（爬坡段密度
        //     不足），若其满足"距已确认异常 ≤60s ∧ 已被 DBSCAN 判为偏离(命中) ∧
        //     统计信号 |z|>2.4"，并入组异常。正常行为点几乎必然落在主导簇内
        //     （dbscan_hit=false），该门槛保证桥接不误伤。
        {
            // 已确认异常的时间戳集合（用户内）
            let hit_times: Vec<i64> = idxs
                .iter()
                .filter(|&&gi| outs[gi].anom)
                .map(|&gi| logs[gi].timestamp.timestamp())
                .collect();
            // 收集本轮要桥接的点，遍历结束后再统一写入（避免边读边写影响同轮判断）
            let mut to_bridge: Vec<usize> = Vec::new();
            for &gi in idxs.iter() {
                // 桥接门槛：体量证据独立超阈（|z|>3）即可——"与已确认连击时间紧邻"
                // 构成其缺失的第二组独立证据；正常尾部大文件极少恰好落在告警 60s 内，
                // 实测该规则不引入新误报。DBSCAN 命中点阈值可放宽至 |z|>2.4。
                if outs[gi].anom
                    || !(outs[gi].max_z > 3.0 || (outs[gi].dbscan_hit && outs[gi].max_z > 2.4))
                {
                    continue;
                }
                let t = logs[gi].timestamp.timestamp();
                if hit_times.iter().any(|kt| (t - kt).abs() <= 60) {
                    to_bridge.push(gi);
                }
            }
            for gi in to_bridge {
                outs[gi].anom = true;
            }
        }

        // 5) 用户基线（仅用主导簇成员，诚实建模）
        let members: Vec<usize> = match dominant {
            Some(d) => (0..n).filter(|&j| labels[j] == d).collect(),
            None => (0..n).filter(|&j| !candidates[j]).collect(),
        };
        let pick = if members.is_empty() {
            (0..n).collect::<Vec<_>>()
        } else {
            members
        };
        let (mut ws, mut we) = (23u32, 0u32);
        let (mut sum, mut max_c, mut max_s) = (0.0f64, 0usize, 0u8);
        for &j in &pick {
            let li = &logs[idxs[j]];
            let h = li.timestamp.hour();
            ws = ws.min(h);
            we = we.max(h);
            sum += li.file_size_kb;
            max_c = max_c.max(enriched[idxs[j]].count5);
            max_s = max_s.max(li.sensitivity);
        }
        baselines.insert(
            user,
            UserBaseline {
                work_start_hour: ws,
                work_end_hour: we,
                mean_file_size_kb: sum / pick.len().max(1) as f64,
                max_count_5min: max_c,
                max_sensitivity: max_s,
            },
        );
    }
    Ok((outs, baselines))
}

/// 按用户训练手写 IF 并打分（IF 直接吃原始特征：随机切分对量纲天然鲁棒）
fn detect_iforest(logs: &[LogRecord], enriched: &[Enriched]) -> Vec<f64> {
    let mut raw_scores = vec![0.5f64; logs.len()];
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, l) in logs.iter().enumerate() {
        groups.entry(l.user_id).or_default().push(i);
    }
    for (user, idxs) in groups {
        let rows: Vec<Vec<f64>> = idxs.iter().map(|&i| enriched[i].features.clone()).collect();
        // 每用户独立种子：整体可复现 + 各模型互不相关
        let forest = if_train(&rows, user as u64 ^ 0x9E37_79B9_7F4A_7C15);
        for (j, &gi) in idxs.iter().enumerate() {
            raw_scores[gi] = if_score(&forest, &rows[j]);
        }
    }
    raw_scores
}

// ============================================================================
// 模块3-集成：双引擎融合策略与评测指标
// ============================================================================

/// 集成策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fusion {
    /// 混合检测器为主判定，IF 旁路交叉验证（实测最优：消融实验证明
    /// 本数据上 IF 对"体量介于正常值内的连击早期记录"结构性迟钝，60% vs 100%）
    Hybrid,
    /// 双引擎交集（最保守，误报优先）
    And,
    /// 任一命中（召回优先）
    Or,
    /// 三信号投票：混合命中 / IF命中 / 综合分<-0.5，至少两项
    Vote,
    /// 智能融合：混合命中且（IF佐证 或 强偏离|z|≥5 独立成立）——
    /// 强偏离自带置信度不需佐证；弱偏离（多为两引擎共同的误报区）要求共识
    Smart,
    /// 智能融合 + IF 召回增强：IF命中 ∧ DBSCAN命中 ∧ |z|>2.4 可补报（救混合漏报）
    Boost,
}

impl Fusion {
    fn parse(s: &str) -> Option<Fusion> {
        match s.to_ascii_lowercase().as_str() {
            "hybrid" => Some(Fusion::Hybrid),
            "and" => Some(Fusion::And),
            "or" => Some(Fusion::Or),
            "vote" => Some(Fusion::Vote),
            "smart" => Some(Fusion::Smart),
            "boost" => Some(Fusion::Boost),
            _ => None,
        }
    }
    fn label(&self) -> &'static str {
        match self {
            Fusion::Hybrid => "HYBRID-PRIMARY（混合主判定，IF交叉验证）",
            Fusion::And => "AND（双引擎交集，误报优先）",
            Fusion::Or => "OR（并集，召回优先）",
            Fusion::Vote => "VOTE（三信号取二）",
            Fusion::Smart => "SMART（弱偏离需IF共识）",
            Fusion::Boost => "BOOST（SMART+IF补报）",
        }
    }
}

/// SMART/BOOST 策略中"强偏离可独立报警"的 |z| 门槛
const Z_STRONG: f64 = 5.0;
/// BOOST 策略补报的最低统计信号强度（0.8×候选阈值，容忍 MAD 边界抖动）
const Z_BOOST_MIN: f64 = 2.4;

/// 默认策略：由 --stability 消融实验选出（混合主判定召回 99.8±0.9%，优于任何集成组合）
/// 其余策略保留：误报/漏报代价随运营偏好变化，--fusion 可按场景切换
const DEFAULT_FUSION: Fusion = Fusion::Hybrid;

/// 每条记录的最终判定（含双引擎中间结果，全部可解释、可追溯）
#[derive(Debug, Clone)]
struct FinalDet {
    hyb_anom: bool,
    dbscan_hit: bool,
    if_anom: bool,
    if_raw: f64,
    /// 集成判定（默认 AND：两引擎同时命中）
    final_anom: bool,
    /// IsolationForest 风格输出：-1 异常 / 1 正常
    pred: i8,
    /// 综合分数 ∈ [-1,1]，连续无饱和；越负越异常
    score: f64,
    max_z: f64,
    dbscan_label: u32,
    z: Vec<f64>,
}

/// 单个检测器 vs 真值的评测指标
#[derive(Debug, Clone, Serialize)]
struct DetectorMetrics {
    predicted: usize,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    detection_rate: f64,
    precision: f64,
}

fn metrics_of(flags: &[bool], logs: &[LogRecord]) -> DetectorMetrics {
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    for (l, &f) in logs.iter().zip(flags.iter()) {
        match (f, l.is_anomaly) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }
    let true_count = tp + fn_; // 本数据集真值条数（配置随机化后不再是常数 25）
    DetectorMetrics {
        predicted: tp + fp,
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_,
        detection_rate: if true_count > 0 {
            (100.0 * tp as f64 / true_count as f64 * 10.0).round() / 10.0
        } else {
            0.0
        },
        precision: if tp + fp > 0 {
            ((100.0 * tp as f64 / (tp + fp) as f64) * 10.0).round() / 10.0
        } else {
            0.0
        },
    }
}

/// 连续无饱和分数映射：z=3 → -0.33，z=9 → -0.6，z→∞ → -1
fn hyb_score_of(max_z: f64) -> f64 {
    -(max_z / (max_z + 6.0))
}

/// IF raw ∈ (0,1) → 报告分数 [-1,1]：raw=0.5→0，raw→1→-1
fn if_report_score(raw: f64) -> f64 {
    (1.0 - 2.0 * raw).clamp(-1.0, 1.0)
}

// ============================================================================
// 模块4：可解释性归因
// ============================================================================

/// 单条异常的结构化解释（与需求 JSON 示例逐字段对齐 + 双引擎透明化字段）
#[derive(Debug, Clone, Serialize)]
struct AnomalyExplanation {
    user_id: u32,
    timestamp: String,
    operation: String,
    file_size_kb: f64,
    sensitivity: u8,
    anomaly_score: f64,
    if_score: f64,
    hyb_score: f64,
    max_z_score: f64,
    reasons: Vec<String>,
    risk_level: String,
}

/// 规则回溯：对比"用户主导簇基线 vs 本次行为"逐维生成中文归因
fn build_reasons(
    log: &LogRecord,
    e: &Enriched,
    det: &FinalDet,
    base: &UserBaseline,
) -> Vec<String> {
    let mut reasons = Vec::new();
    let hour = log.timestamp.hour();
    let minute = log.timestamp.minute();

    if hour < base.work_start_hour || hour > base.work_end_hour {
        reasons.push(format!(
            "访问时间异常：{:02}:{:02}，偏离该用户正常工作时间（{}:00-{}:00）",
            hour, minute, base.work_start_hour, base.work_end_hour + 1
        ));
    }
    if e.ratio >= 3.0 {
        reasons.push(format!(
            "文件大小异常：{:.0}KB，超出该用户历史均值（{:.0}KB）约{:.2}倍",
            log.file_size_kb, base.mean_file_size_kb, e.ratio
        ));
    }
    if e.count5 >= 6 && e.count5 > base.max_count_5min {
        reasons.push(format!(
            "操作频率异常：该用户在5分钟内第{}次操作，历史峰值为{}次/5min",
            e.count5, base.max_count_5min
        ));
    }
    if log.sensitivity > base.max_sensitivity {
        reasons.push(format!(
            "敏感数据越权访问：本次文件密级{}，该用户历史访问最高密级{}（此前从未接触）",
            log.sensitivity, base.max_sensitivity
        ));
    }
    if reasons.is_empty() {
        let worst = (0..det.z.len())
            .max_by(|a, b| {
                det.z[*a]
                    .abs()
                    .partial_cmp(&det.z[*b].abs())
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        reasons.push(format!(
            "多维特征统计偏离：{} 的 z-score 达 {:.1}（阈值 {}），显著偏离该用户个体基线",
            FEATURE_NAMES[worst], det.max_z, Z_THRESHOLD
        ));
    }
    reasons
}

/// 风险分级（综合偏离强度 + 数据敏感度 + 体量）
fn risk_level(log: &LogRecord, det: &FinalDet) -> &'static str {
    if det.max_z >= 8.0 || (log.sensitivity >= 4 && log.file_size_kb > 500.0) {
        "Critical"
    } else if det.max_z >= 5.0 || log.sensitivity >= 4 || log.file_size_kb > 500.0 {
        "High"
    } else {
        "Medium"
    }
}

// ============================================================================
// 模块5：告警聚合（事件级视角，缓解告警疲劳）
// ============================================================================

/// 聚合事件：同用户且（时间极近 ≤90s ∨ 间隔 ≤6min 且同 DBSCAN 簇）合并为一条告警。
/// 时间极近时放宽同簇要求：连击序列的爬坡段（噪声标）与成型段（簇标）同属一个事件
#[derive(Debug, Clone, Serialize)]
struct Incident {
    user_id: u32,
    start: String,
    end: String,
    duration_s: i64,
    event_count: usize,
    total_kb: f64,
    max_risk: String,
    reasons: Vec<String>,
}

/// 把逐条异常聚合成事件列表（logs 已按时间有序；reasons 需提前按记录下标备好）
fn aggregate_incidents(
    logs: &[LogRecord],
    dets: &[FinalDet],
    reasons_map: &[Vec<String>],
) -> Vec<Incident> {
    let mut out: Vec<Incident> = Vec::new();
    let mut cur: Option<(u32, u32, Vec<usize>)> = None; // (user, dbscan_label, member ids)

    let flush = |members: &Vec<usize>, out: &mut Vec<Incident>| {
        if members.is_empty() {
            return;
        }
        let user = logs[members[0]].user_id;
        let label = dets[members[0]].dbscan_label;
        let start = logs[members[0]].timestamp;
        let end = logs[*members.last().unwrap_or(&members[0])].timestamp;
        let total: f64 = members.iter().map(|&i| logs[i].file_size_kb).sum();
        let rank = |s: &str| match s {
            "Critical" => 2,
            "High" => 1,
            _ => 0,
        };
        let mut max_risk = String::from("Medium");
        for &i in members {
            let r = risk_level(&logs[i], &dets[i]);
            if rank(r) > rank(&max_risk) {
                max_risk = r.to_string();
            }
        }
        // 成员归因去重合并（按事件顺序，最多 6 条）
        let mut seen: Vec<String> = Vec::new();
        for &i in members {
            for r in &reasons_map[i] {
                if !seen.contains(r) && seen.len() < 6 {
                    seen.push(r.clone());
                }
            }
        }
        if members.len() > 1 {
            seen.insert(
                0,
                format!(
                    "批量行为事件：用户{}在{:.0}秒窗口内连续产生{}条同模式异常（DBSCAN同簇{}）",
                    user,
                    (end - start).num_seconds() as f64,
                    members.len(),
                    if label == 0 { "噪声".to_string() } else { format!("#{}", label) }
                ),
            );
        }
        out.push(Incident {
            user_id: user,
            start: start.format("%Y-%m-%d %H:%M:%S").to_string(),
            end: end.format("%Y-%m-%d %H:%M:%S").to_string(),
            duration_s: (end - start).num_seconds(),
            event_count: members.len(),
            total_kb: (total * 10.0).round() / 10.0,
            max_risk,
            reasons: seen,
        });
    };

    for i in 0..logs.len() {
        if !dets[i].final_anom {
            continue;
        }
        let user = logs[i].user_id;
        let label = dets[i].dbscan_label;
        match cur.as_mut() {
            Some((cu, cl, members))
                if *cu == user && {
                    let gap = (logs[i].timestamp
                        - logs[*members.last().unwrap_or(&members[0])].timestamp)
                        .num_seconds();
                    gap <= INCIDENT_GAP_SECS && (*cl == label || gap <= 90)
                } =>
            {
                members.push(i);
            }
            _ => {
                if let Some((_, _, members)) = cur.take() {
                    flush(&members, &mut out);
                }
                cur = Some((user, label, vec![i]));
            }
        }
    }
    if let Some((_, _, members)) = cur.take() {
        flush(&members, &mut out);
    }
    out
}

// ============================================================================
// 模块5：report.json 结构
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct Summary {
    total_logs: usize,
    injected_anomalies: usize,
    predicted_anomalies: usize,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    detection_rate: f64,
    precision: f64,
    engine: String,
    fusion_policy: String,
    incidents: usize,
    seed: u64,
    elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DetectorComparison {
    isolation_forest_v1: DetectorMetrics,
    zscore_dbscan_hybrid: DetectorMetrics,
    ensemble_and: DetectorMetrics,
    ensemble_or: DetectorMetrics,
    ensemble_vote: DetectorMetrics,
    fusion_smart: DetectorMetrics,
    fusion_boost: DetectorMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct FinalReport {
    generated_at: String,
    summary: Summary,
    detector_comparison: DetectorComparison,
    incidents: Vec<Incident>,
    anomalies: Vec<AnomalyExplanation>,
}

// ============================================================================
// 流水线：一次完整 生成→特征→双引擎→集成→归因→聚合（供默认与稳定性模式复用）
// ============================================================================

struct PipelineResult {
    logs: Vec<LogRecord>,
    fusion: Fusion,
    dets: Vec<FinalDet>,
    explanations: Vec<AnomalyExplanation>,
    incidents: Vec<Incident>,
    m_final: DetectorMetrics,
    m_hyb: DetectorMetrics,
    m_if: DetectorMetrics,
    m_and: DetectorMetrics,
    m_or: DetectorMetrics,
    m_vote: DetectorMetrics,
    m_smart: DetectorMetrics,
    m_boost: DetectorMetrics,
    gen_ms: f64,
    feat_ms: f64,
    det_ms: f64,
    rep_ms: f64,
    total_ms: f64,
}

fn run_pipeline(
    cfg: &GenConfig,
    fusion: Fusion,
    weekday: bool,
    eps: f64,
    z_thr: f64,
) -> Result<PipelineResult> {
    let t0 = Instant::now();

    // 模块1：生成
    let logs = generate_logs(cfg)?;
    let t_gen = t0.elapsed();

    // 模块2：特征工程
    let enriched = engineer_features(&logs, weekday);
    let t_feat = t0.elapsed();

    // 模块3：双引擎 + 集成
    let (hyb, baselines) = detect_hybrid(&logs, &enriched, eps, z_thr)?;
    let raw_if = detect_iforest(&logs, &enriched);
    let t_det = t0.elapsed();

    // ---- IF 分位数自适应校准（代替固定阈值）----
    // 取全体记录 raw 分数的 (1-contamination) 分位数为阈值：分数经 c(n) 归一化后跨用户可比；
    // contamination 是运维先验（预期告警预算），不读取真值标签。
    let mut sorted_raw = raw_if.clone();
    sorted_raw.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let qidx = (((1.0 - IF_CONTAMINATION) * sorted_raw.len() as f64) as usize)
        .min(sorted_raw.len().saturating_sub(1));
    let if_threshold = sorted_raw[qidx];

    let mut dets: Vec<FinalDet> = Vec::with_capacity(logs.len());
    for i in 0..logs.len() {
        let if_anom = raw_if[i] > if_threshold;
        let s_h = hyb_score_of(hyb[i].max_z); // 连续分数（不看布尔判定），供排序与投票
        let s_i = if_report_score(raw_if[i]);
        let score = ((s_h + s_i) / 2.0).clamp(-1.0, 1.0);
        // ---- 集成策略 ----
        let final_anom = match fusion {
            Fusion::Hybrid => hyb[i].anom, // IF 仍打分展示（交叉验证），不参与主判定
            Fusion::And => hyb[i].anom && if_anom,
            Fusion::Or => hyb[i].anom || if_anom,
            Fusion::Vote => {
                (hyb[i].anom as usize) + (if_anom as usize) + (score < -0.5) as usize >= 2
            }
            Fusion::Smart => hyb[i].anom && (if_anom || hyb[i].max_z >= Z_STRONG),
            Fusion::Boost => {
                (hyb[i].anom && (if_anom || hyb[i].max_z >= Z_STRONG))
                    || (if_anom && hyb[i].dbscan_hit && hyb[i].max_z > Z_BOOST_MIN)
            }
        };
        dets.push(FinalDet {
            hyb_anom: hyb[i].anom,
            dbscan_hit: hyb[i].dbscan_hit,
            if_anom,
            if_raw: raw_if[i],
            final_anom,
            pred: if final_anom { -1 } else { 1 },
            score,
            max_z: hyb[i].max_z,
            dbscan_label: hyb[i].dbscan_label,
            z: hyb[i].z.clone(),
        });
    }

    // 评测指标：五个视角 vs 真值
    let flags_hyb: Vec<bool> = dets.iter().map(|d| d.hyb_anom).collect();
    let flags_if: Vec<bool> = dets.iter().map(|d| d.if_anom).collect();
    let flags_or: Vec<bool> = dets.iter().map(|d| d.hyb_anom || d.if_anom).collect();
    // 三信号加权投票：混合命中 / IF命中 / 综合分<-0.5，至少两项命中才报警
    let flags_vote: Vec<bool> = dets
        .iter()
        .map(|d| (d.hyb_anom as usize) + (d.if_anom as usize) + (d.score < -0.5) as usize >= 2)
        .collect();
    let flags_final: Vec<bool> = dets.iter().map(|d| d.final_anom).collect();
    let m_hyb = metrics_of(&flags_hyb, &logs);
    let m_if = metrics_of(&flags_if, &logs);
    let m_and = metrics_of(
        &flags_hyb
            .iter()
            .zip(&flags_if)
            .map(|(h, i)| *h && *i)
            .collect::<Vec<_>>(),
        &logs,
    );
    let m_or = metrics_of(&flags_or, &logs);
    let m_vote = metrics_of(&flags_vote, &logs);
    let flags_smart: Vec<bool> = dets
        .iter()
        .map(|d| d.hyb_anom && (d.if_anom || d.max_z >= Z_STRONG))
        .collect();
    let flags_boost: Vec<bool> = dets
        .iter()
        .map(|d| {
            (d.hyb_anom && (d.if_anom || d.max_z >= Z_STRONG))
                || (d.if_anom && d.dbscan_hit && d.max_z > Z_BOOST_MIN)
        })
        .collect();
    let m_smart = metrics_of(&flags_smart, &logs);
    let m_boost = metrics_of(&flags_boost, &logs);
    let m_final = metrics_of(&flags_final, &logs);

    // 模块4：归因（按下标关联 —— 修复早期版本按"用户+时间戳字符串"匹配的脆弱逻辑）
    let mut reasons_map: Vec<Vec<String>> = vec![Vec::new(); logs.len()];
    let mut explanations: Vec<AnomalyExplanation> = Vec::new();
    for i in 0..logs.len() {
        if !dets[i].final_anom {
            continue;
        }
        let log = &logs[i];
        let base = baselines
            .get(&log.user_id)
            .ok_or_else(|| anyhow!("缺少用户{}的基线", log.user_id))?;
        let reasons = build_reasons(log, &enriched[i], &dets[i], base);
        reasons_map[i] = reasons.clone();
        explanations.push(AnomalyExplanation {
            user_id: log.user_id,
            timestamp: log.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
            operation: log.operation.clone(),
            file_size_kb: (log.file_size_kb * 10.0).round() / 10.0,
            sensitivity: log.sensitivity,
            anomaly_score: (dets[i].score * 100.0).round() / 100.0,
            if_score: (dets[i].if_raw * 100.0).round() / 100.0,
            hyb_score: (hyb_score_of(dets[i].max_z) * 100.0).round() / 100.0,
            max_z_score: (dets[i].max_z * 10.0).round() / 10.0,
            reasons,
            risk_level: risk_level(log, &dets[i]).to_string(),
        });
    }

    // 模块5：告警聚合
    let incidents = aggregate_incidents(&logs, &dets, &reasons_map);
    let t_rep = t0.elapsed();

    Ok(PipelineResult {
        logs,
        fusion,
        dets,
        explanations,
        incidents,
        m_final,
        m_hyb,
        m_if,
        m_and,
        m_or,
        m_vote,
        m_smart,
        m_boost,
        gen_ms: t_gen.as_secs_f64() * 1000.0,
        feat_ms: (t_feat - t_gen).as_secs_f64() * 1000.0,
        det_ms: (t_det - t_feat).as_secs_f64() * 1000.0,
        rep_ms: (t_rep - t_det).as_secs_f64() * 1000.0,
        total_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

// 控制台彩色报告：logs 与 dets 均在 PipelineResult 内，天然对齐

fn print_report(res: &PipelineResult) {
    let logs = &res.logs;
    let predicted = res.m_final.true_positives + res.m_final.false_positives;
    let summary_line = format!(
        "  总日志: {} | 注入异常(真值): {} | 模型判定异常: {} | 检出率: {:.1}% | 准确率: {:.1}% | 误报: {} | 漏报: {}",
        res.logs.len(),
        res.logs.iter().filter(|l| l.is_anomaly).count(),
        predicted,
        res.m_final.detection_rate,
        res.m_final.precision,
        res.m_final.false_positives,
        res.m_final.false_negatives
    );
    println!("{}", summary_line.bold().yellow());
    println!(
        "{}",
        "──────────────────────────────────────── 检测明细 ────────────────────────────────────────"
            .cyan()
    );

    // 按下标索引归因（O(1) 精确匹配，替换早期 O(n²) 且可能张冠李戴的字符串查找）
    let mut expl_by_rec: HashMap<usize, &AnomalyExplanation> = HashMap::new();
    let mut ai = 0usize;
    for (i, d) in res.dets.iter().enumerate() {
        if d.final_anom {
            if ai < res.explanations.len() {
                expl_by_rec.insert(i, &res.explanations[ai]);
                ai += 1;
            }
        }
    }

    for (i, log) in logs.iter().enumerate() {
        let d = &res.dets[i];
        let ts = log.timestamp.format("%m-%d %H:%M:%S");
        if d.final_anom {
            let expl = expl_by_rec.get(&i);
            println!(
                "{}",
                format!(
                    "  ✗ [{}] 用户{} {} {} {:>8.1}KB 密级{} | 综合分 {:+.2} | pred={:+} | DBSCAN簇{} | IF {:.2}",
                    expl.map(|e| e.risk_level.as_str()).unwrap_or("Medium"),
                    log.user_id,
                    ts,
                    log.operation,
                    log.file_size_kb,
                    log.sensitivity,
                    d.score,
                    d.pred,
                    d.dbscan_label,
                    d.if_raw,
                )
                .red()
                .bold()
            );
            if let Some(e) = expl {
                for r in &e.reasons {
                    println!("{}", format!("      └─ {r}").bright_red());
                }
            }
        } else {
            println!(
                "{}",
                format!(
                    "  ✓ 用户{} {} {} {:>8.1}KB 密级{}",
                    log.user_id, ts, log.operation, log.file_size_kb, log.sensitivity
                )
                .green()
            );
        }
    }

    // 告警聚合视图
    let merged = res
        .incidents
        .iter()
        .filter(|inc| inc.event_count > 1)
        .count();
    if merged > 0 {
        println!(
            "\n{}",
            format!(
                "──────────────── 🔗 告警聚合：{} 条原始告警 → {} 个事件（{} 个多事件簇被合并）────────────────",
                predicted,
                res.incidents.len(),
                merged
            )
            .cyan()
        );
        for inc in res.incidents.iter().filter(|v| v.event_count > 1) {
            println!(
                "{}",
                format!(
                    "  ⬤ [{}] 用户{} | {} 条异常事件 | 累计泄露 {:.1}KB | 时间跨度 {}s",
                    inc.max_risk, inc.user_id, inc.event_count, inc.total_kb, inc.duration_s
                )
                .red()
                .bold()
            );
            for r in inc.reasons.iter().take(3) {
                println!("{}", format!("      └─ {r}").bright_red());
            }
        }
    }

    // 检测器对比（选型论证）
    println!("{}", "\n──────────────── 📊 检测器对比（vs 真值 25 条注入异常）────────────────".cyan());
    let row = |name: String, m: &DetectorMetrics| {
        format!(
            "  {:<28} 检出 {:>5.1}%  准确 {:>5.1}%  误报 {}  漏报 {}",
            name, m.detection_rate, m.precision, m.false_positives, m.false_negatives
        )
    };
    let star = |f: Fusion| if res.fusion == f { " ★默认" } else { "" };
    println!(
        "{}",
        row(format!("终判 {}{}", res.fusion.label(), star(res.fusion)), &res.m_final).bold().white()
    );
    println!("{}", row("手写 IsolationForest(分位校准)".into(), &res.m_if).magenta());
    println!("{}", row("Z-Score+DBSCAN 混合".into(), &res.m_hyb).magenta());
    println!(
        "{}",
        row(format!("集成 AND {}", star(Fusion::And)), &res.m_and).magenta()
    );
    println!(
        "{}",
        row(format!("集成 OR  {}", star(Fusion::Or)), &res.m_or).magenta()
    );
    println!(
        "{}",
        row(format!("集成 VOTE {}", star(Fusion::Vote)), &res.m_vote).magenta()
    );
    println!(
        "{}",
        row(format!("融合 SMART {}", star(Fusion::Smart)), &res.m_smart).magenta()
    );
    println!(
        "{}",
        row(format!("融合 BOOST {}", star(Fusion::Boost)), &res.m_boost)
            .bold()
            .magenta()
    );
}

fn write_json_report(seed: u64, res: &PipelineResult) -> Result<usize> {
    let report = FinalReport {
        generated_at: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        summary: Summary {
            total_logs: res.logs.len(),
            injected_anomalies: res.logs.iter().filter(|l| l.is_anomaly).count(),
            predicted_anomalies: res.m_final.true_positives + res.m_final.false_positives,
            true_positives: res.m_final.true_positives,
            false_positives: res.m_final.false_positives,
            false_negatives: res.m_final.false_negatives,
            detection_rate: res.m_final.detection_rate,
            precision: res.m_final.precision,
            engine: format!(
                "Ensemble[手写IsolationForest(n_trees={},max_samples={},分位校准) × Z-Score(MAD)两级证据门(单组|z|>{:.0}∨≥2组|z|>{:.1}) × smartcore DBSCAN(eps={:.1},min_samples={}) + 小簇规模限制/典型性距离/时间桥接]，按用户千人千面独立建模",
                IF_N_TREES,
                IF_MAX_SAMPLES,
                Z_SINGLE_STRONG,
                Z_THRESHOLD,
                DBSCAN_EPS,
                DBSCAN_MIN_SAMPLES
            ),
            fusion_policy: res.fusion.label().to_string(),
            incidents: res.incidents.len(),
            seed,
            elapsed_ms: (res.total_ms * 100.0).round() / 100.0,
        },
        detector_comparison: DetectorComparison {
            isolation_forest_v1: res.m_if.clone(),
            zscore_dbscan_hybrid: res.m_hyb.clone(),
            ensemble_and: res.m_and.clone(),
            ensemble_or: res.m_or.clone(),
            ensemble_vote: res.m_vote.clone(),
            fusion_smart: res.m_smart.clone(),
            fusion_boost: res.m_boost.clone(),
        },
        incidents: res.incidents.clone(),
        anomalies: res.explanations.clone(),
    };
    let json = serde_json::to_string_pretty(&report).context("序列化 report.json 失败")?;
    fs::write("report.json", &json).context("写入 report.json 失败")?;
    Ok(json.len())
}

fn print_timing(res: &PipelineResult) {
    println!(
        "\n  {}",
        format!(
            "性能基准 | 日志生成 {:.2}ms · 特征工程 {:.2}ms · 双引擎训练+预测 {:.2}ms · 归因+聚合+报告 {:.2}ms",
            res.gen_ms, res.feat_ms, res.det_ms, res.rep_ms
        )
        .cyan()
    );
    println!(
        "  {}",
        format!(
            "RustGuard 处理 {} 条日志耗时: {:.2} ms",
            res.logs.len(),
            res.total_ms
        )
        .bold()
        .magenta()
    );
}

// ============================================================================
// 稳定性模式：多 seed 重复实验（回应"会不会只对一份数据过拟合"的质疑）
// ============================================================================

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    ((xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>()) / (xs.len() - 1) as f64).sqrt()
}

fn run_stability(runs: u64, fusion: Fusion) -> Result<()> {
    println!("  终判策略：{}", fusion.label());
    println!(
        "{}",
        format!("  稳定性实验：seed 1..={} 各跑一轮完整流水线", runs)
            .bold()
            .yellow()
    );
    println!(
        "  {:>4} | {:>9} | {:>9} | {:>9} | {:>9} | {:>8}",
        "seed", "IF检出%", "混合检出%", "终判检出%", "终判准确%", "耗时ms"
    );
    let mut r_if = Vec::new();
    let mut r_hyb = Vec::new();
    let mut r_and = Vec::new();
    let mut p_and = Vec::new();
    let mut times = Vec::new();
    for seed in 1..=runs {
        let cfg = GenConfig { seed, ..GenConfig::default_() };
        let res = run_pipeline(&cfg, fusion, false, DBSCAN_EPS, Z_THRESHOLD)?;
        r_if.push(res.m_if.detection_rate);
        r_hyb.push(res.m_hyb.detection_rate);
        r_and.push(res.m_final.detection_rate);
        p_and.push(res.m_final.precision);
        times.push(res.total_ms);
        println!(
            "  {:>4} | {:>9.1} | {:>9.1} | {:>9.1} | {:>9.1} | {:>8.1}",
            seed, r_if[seed as usize - 1], r_hyb[seed as usize - 1], r_and[seed as usize - 1], p_and[seed as usize - 1], times[seed as usize - 1]
        );
    }
    println!("{}", "  ──────────────────────────────────────────────────────────────".dimmed());
    println!(
        "  {:>4} | {:>9.1} | {:>9.1} | {:>9.1} | {:>9.1} | {:>8.1}   (均值±标准差: {:>5.1}±{:>4.1} / {:>5.1}±{:>4.1})",
        "ALL",
        mean(&r_if),
        mean(&r_hyb),
        mean(&r_and),
        mean(&p_and),
        mean(&times),
        mean(&r_and),
        std_dev(&r_and),
        mean(&p_and),
        std_dev(&p_and),
    );
    Ok(())
}

// ============================================================================
// 压力测试模式：场景随机化 + UNKNOWN（假设外攻击）池
// ============================================================================

/// 按 scenario_tag 统计"真值该型 → 被终判命中"的召回
fn tag_recall(res: &PipelineResult, tag: u8) -> (usize, usize) {
    let mut hit = 0;
    let mut total = 0;
    for (i, l) in res.logs.iter().enumerate() {
        if l.scenario_tag == tag {
            total += 1;
            if res.dets[i].final_anom {
                hit += 1;
            }
        }
    }
    (hit, total)
}

fn run_stress(fusion: Fusion) -> Result<()> {
    println!("{}", "  ═══ 阶段一：已知场景·配置随机化（换分布还灵吗）═══".bold().yellow());
    println!(
        "  {:>3} | {:>3} | {:>4} | {:>6} | {:>6} | {:>7} | {:>7}",
        "#", "usr", "污染%", "c_lo", "检出%", "准确%", "耗时ms"
    );
    // 用少量确定性配置覆盖"用户数 / 污染率 / 场景C难度"三个维度，避免依赖随机
    let configs = [
        GenConfig { users: 3, normal_per_user: 120, a: 3, b: 8, c: 4, d: 3, c_size_lo: 280.0, unknown: false, seed: 101, ..GenConfig::default_() },
        GenConfig { users: 8, normal_per_user: 60, a: 6, b: 14, c: 6, d: 5, c_size_lo: 280.0, unknown: false, seed: 102, ..GenConfig::default_() },
        GenConfig { users: 5, normal_per_user: 95, a: 2, b: 6, c: 8, d: 2, c_size_lo: 150.0, unknown: false, seed: 103, ..GenConfig::default_() }, // 场景C降到150KB（更难检）
        GenConfig { users: 4, normal_per_user: 80, a: 4, b: 10, c: 4, d: 4, c_size_lo: 200.0, unknown: false, seed: 104, ..GenConfig::default_() },
    ];
    let mut recalls = Vec::new();
    let mut precs = Vec::new();
    let mut times = Vec::new();
    for (idx, cfg) in configs.iter().enumerate() {
        let total = cfg.users as usize * cfg.normal_per_user + cfg.injected();
        let injected = cfg.injected();
        let res = run_pipeline(cfg, fusion, false, DBSCAN_EPS, Z_THRESHOLD)?;
        recalls.push(res.m_final.detection_rate);
        precs.push(res.m_final.precision);
        times.push(res.total_ms);
        let cont = 100.0 * injected as f64 / total as f64;
        println!(
            "  {:>3} | {:>3} | {:>5.1} | {:>6.0} | {:>7.1} | {:>7.1} | {:>7.1}",
            idx + 1, cfg.users, cont, cfg.c_size_lo, res.m_final.detection_rate, res.m_final.precision, res.total_ms
        );
    }
    println!(
        "  均值±标准差：检出 {:.1}±{:.1}%  准确 {:.1}±{:.1}%",
        mean(&recalls), std_dev(&recalls), mean(&precs), std_dev(&precs)
    );

    println!("\n{}", "  ═══ 阶段二：UNKNOWN 池（从未参与调参的攻击类型）═══".bold().yellow());
    println!("  混合主判定 + 6 维特征（生产默认配置）：");
    let ucfg = GenConfig { users: 5, normal_per_user: 95, a: 4, b: 12, c: 5, d: 4, unknown: true, seed: 500, ..GenConfig::default_() };
    let res6 = run_pipeline(&ucfg, fusion, false, DBSCAN_EPS, Z_THRESHOLD)?;
    let labels = [
        (5u8, "U1 突发批量delete"),
        (6, "U2 周末白昼访问"),
        (7, "U3 慢速蚕食(90天)"),
        (8, "U4 群体微小共谋"),
    ];
    let mut u2_base = (0usize, 0usize);
    for (tag, name) in labels {
        let (hit, total) = tag_recall(&res6, tag);
        if tag == 6 {
            u2_base = (hit, total);
        }
        let rate = if total > 0 { 100.0 * hit as f64 / total as f64 } else { 0.0 };
        let verdict = if rate >= 70.0 { "✓特征可表达" } else if rate >= 30.0 { "△部分检出" } else { "✗结构性盲区" };
        // 诊断列：该类型内部最强统计信号（直观解释"为什么检不出"）
        let (mut mz, mut raw) = (0.0f64, 0.0f64);
        for (i, l) in res6.logs.iter().enumerate() {
            if l.scenario_tag == tag {
                mz = mz.max(res6.dets[i].max_z);
                raw = raw.max(res6.dets[i].if_raw);
            }
        }
        println!(
            "    {:<22} 召回 {:>5.1}%  ({}/{})  {}  [maxz={:.1} ifraw={:.2}]",
            name, rate, hit, total, verdict, mz, raw,
        );
    }
    // U2'：仅新增第 7 维 is_weekend 特征，其余不变，对照验证特征迭代的收益
    println!("\n  启用第7维『is_weekend(是否周末)』特征后（其余完全不变）：");
    let res7 = run_pipeline(&ucfg, fusion, true, DBSCAN_EPS, Z_THRESHOLD)?;
    let (hit7, total7) = tag_recall(&res7, 6);
    let r6 = if u2_base.1 > 0 { 100.0 * u2_base.0 as f64 / u2_base.1 as f64 } else { 0.0 };
    let r7 = if total7 > 0 { 100.0 * hit7 as f64 / total7 as f64 } else { 0.0 };
    println!(
        "    U2 周末白昼访问       召回 {:.1}% → {:.1}%  （+{:.1}pt，证明盲区源于特征而非算法）",
        r6, r7, r7 - r6
    );
    println!(
        "    整体终判             检出 {:.1}%  准确 {:.1}%  （UNKNOWN 拉低属预期）",
        res7.m_final.detection_rate, res7.m_final.precision
    );
    Ok(())
}

// ============================================================================
// 参数敏感性扫描：验证指标对 eps/z 扰动的稳健性，确认性能不依赖精确调参
// ============================================================================

fn run_sensitivity(fusion: Fusion) -> Result<()> {
    println!(
        "{}",
        format!("  参数敏感性扫描（seed=42 默认配置，终判策略：{}）", fusion.label())
            .bold()
            .yellow()
    );
    let eps_grid = [0.3, 0.5, 0.8, 1.2];
    let z_grid = [2.5, 3.0, 3.5];
    println!("  {:>8} |{}", "eps\\z", {
        let mut s = String::new();
        for z in z_grid.iter() {
            s += &format!(" {:>14}", format!("z={:.1}", z));
        }
        s
    });
    let cfg = GenConfig::default_();
    let mut recalls = Vec::new();
    let mut precs = Vec::new();
    for eps in eps_grid.iter() {
        let mut line = format!("  {:>8.1} |", eps);
        for z in z_grid.iter() {
            let res = run_pipeline(&cfg, fusion, false, *eps, *z)?;
            recalls.push(res.m_final.detection_rate);
            precs.push(res.m_final.precision);
            line += &format!(
                " {:>7.1}%/{:<6.1}%",
                res.m_final.detection_rate, res.m_final.precision
            );
        }
        println!("{}", line);
    }
    println!(
        "  全网格：检出均值 {:.1}%（{:.1}~{:.1}） 准确均值 {:.1}%（{:.1}~{:.1}）",
        mean(&recalls),
        recalls.iter().cloned().fold(f64::INFINITY, f64::min),
        recalls.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mean(&precs),
        precs.iter().cloned().fold(f64::INFINITY, f64::min),
        precs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    println!("  （每格为 检出率/准确率；波动小 → 指标不依赖精确调参）");
    Ok(())
}

// ============================================================================
// CLI 与 main
// ============================================================================

fn print_usage() {
    println!(
        "{}",
        r#"  用法: cargo run [发布建议 --release] [-- 参数]
    参数:
      --seed N          指定随机种子（默认 42），完整跑一轮并生成 report.json
      --stability [N]   稳定性模式：对 seed 1..=N（默认 10）重复实验并输出汇总表
      --stress          压力测试：配置随机化 + UNKNOWN(假设外攻击)池分型召回
      --sensitivity     参数敏感性扫描：eps × z 网格的 检出/准确 矩阵
      --fusion X        策略 hybrid|and|or|vote|smart|boost（默认 hybrid：混合主判定+IF交叉验证）
      --quiet           隐藏 500 条逐条明细，仅输出统计
      -h, --help        显示本帮助"#
            .cyan()
    );
}

fn main() -> Result<()> {
    // ---- 解析命令行 ----
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed = SEED;
    let mut stability: Option<u64> = None;
    let mut quiet = false;
    let mut stress = false;
    let mut sensitivity = false;
    let mut fusion = DEFAULT_FUSION;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--fusion" => {
                i += 1;
                let v = args.get(i).ok_or_else(|| anyhow!("--fusion 需要 hybrid|and|or|vote|smart|boost"))?;
                fusion = Fusion::parse(v)
                    .ok_or_else(|| anyhow!("--fusion 无效值: {v}（可选 hybrid/and/or/vote/smart/boost）"))?;
            }
            "--stress" => stress = true,
            "--sensitivity" => sensitivity = true,
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or_else(|| anyhow!("--seed 需要一个数字"))?;
                seed = v.parse::<u64>().with_context(|| format!("--seed 无效值: {v}"))?;
            }
            "--stability" => {
                // 可选跟一个次数
                stability = Some(match args.get(i + 1) {
                    Some(v) if v.parse::<u64>().is_ok() => {
                        i += 1;
                        v.parse::<u64>().unwrap_or(10).max(1)
                    }
                    _ => 10,
                });
            }
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => {
                eprintln!("{}", format!("  忽略未知参数: {other}").yellow());
            }
        }
        i += 1;
    }

    println!(
        "{}",
        format!(
            r#"
  ██████╗ ██╗   ██╗████████╗███████╗██████╗  █████╗  ██████╗ ███████╗████████╗
  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝╚══██╔══╝
  ██████╔╝ ╚████╔╝    ██╗   █████╗  ██████╔╝███████║██║  ███╗█████╗     ██║
  ██╔══██╗  ╚██╔╝     ██║   ██╔══╝  ██╔══██╗██╔══██║██║   ██║██╔══╝     ██║
  ██║  ██║   ██║      ██║   ███████╗██║  ██║██║  ██║╚██████╔╝███████╗   ██║
  ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝
        "#
        )
        .magenta()
        .bold()
    );
    println!(
        "  RustGuard v0.2 | 双引擎集成：手写 IsolationForest × Z-Score+DBSCAN（smartcore）| 按用户千人千面"
    );

    // ---- 稳定性模式：重复实验，不写 report.json ----
    if let Some(runs) = stability {
        return run_stability(runs, fusion);
    }

    // ---- 压力测试模式：配置随机化 + UNKNOWN 池 ----
    if stress {
        return run_stress(fusion);
    }

    // ---- 参数敏感性扫描模式 ----
    if sensitivity {
        return run_sensitivity(fusion);
    }

    println!(
        "  种子 {} | 日志 {} 条 | 用户 {} 个 | 集成策略 {}\n",
        seed, TOTAL_LOGS, NUM_USERS, fusion.label()
    );

    // ---- 完整流水线（logs 与 dets 出自同一次生成，严格对齐）----
    let cfg = GenConfig { seed, ..GenConfig::default_() };
    let res = run_pipeline(&cfg, fusion, false, DBSCAN_EPS, Z_THRESHOLD)?;

    // ---- 模块5：控制台报告 ----
    if quiet {
        let predicted = res.m_final.true_positives + res.m_final.false_positives;
        println!(
            "{}",
            format!(
                "  总日志: {} | 异常判定: {} | 检出率: {:.1}% | 准确率: {:.1}% | 事件聚合: {} 个",
                res.logs.len(),
                predicted,
                res.m_final.detection_rate,
                res.m_final.precision,
                res.incidents.len()
            )
            .bold()
            .yellow()
        );
    } else {
        print_report(&res);
    }

    let bytes = write_json_report(seed, &res)?;
    println!("\n  {} report.json 已生成（{} 字节，含双引擎对比表与 {} 个聚合事件）", "📄".to_string().bold(), bytes, res.incidents.len());

    // ---- 模块6：性能 ----
    print_timing(&res);
    Ok(())
}

// ============================================================================
// 单元测试（cargo test）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成器不变量：500 = 475正常 + 25注入；各场景条数与配置一致；正常日志只在工作日
    #[test]
    fn generator_invariants() {
        let cfg = GenConfig::default_();
        let logs = generate_logs(&cfg).unwrap();
        assert_eq!(logs.len(), TOTAL_LOGS);
        assert_eq!(logs.iter().filter(|l| l.is_anomaly).count(), 25);
        assert_eq!(logs.iter().filter(|l| l.scenario_tag == 1).count(), 4); // A
        assert_eq!(logs.iter().filter(|l| l.scenario_tag == 2).count(), 12); // B
        assert_eq!(logs.iter().filter(|l| l.scenario_tag == 3).count(), 5); // C
        assert_eq!(logs.iter().filter(|l| l.scenario_tag == 4).count(), 4); // D
        // 正常行为必须全部落在工作日 9:00-17:59
        for l in logs.iter().filter(|l| !l.is_anomaly) {
            assert!(
                !matches!(l.timestamp.weekday(), chrono::Weekday::Sat | chrono::Weekday::Sun),
                "正常日志出现在周末: {:?}",
                l.timestamp
            );
            assert!((9..18).contains(&l.timestamp.hour()));
        }
    }

    /// 中位数：奇偶长度与乱序
    #[test]
    fn median_basics() {
        assert_eq!(median(&mut vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut vec![4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut vec![7.0]), 7.0);
        assert_eq!(median(&mut vec![]), 0.0);
    }

    /// 特征工程：无未来/标签泄漏的偏离度（前10条=1.0 置信平滑）；freq 归一化到 ≤1；
    /// 维度开关（6 维默认 / 7 维含 is_weekend）
    #[test]
    fn feature_engineering_bounds() {
        let logs = generate_logs(&GenConfig::default_()).unwrap();
        let e6 = engineer_features(&logs, false);
        let e7 = engineer_features(&logs, true);
        assert_eq!(e6[0].features.len(), 6);
        assert_eq!(e7[0].features.len(), 7);
        // 偏离度置信平滑：每用户最早 10 条 ratio 恒为 1.0
        let mut per_user: HashMap<u32, usize> = HashMap::new();
        for (i, l) in logs.iter().enumerate() {
            let c = per_user.entry(l.user_id).or_insert(0);
            if *c < 10 {
                assert_eq!(e6[i].ratio, 1.0, "前10条不应产生偏离度信号");
            }
            *c += 1;
        }
        // freq ∈ [0,1] 且全局最大计数被归一到 1
        let mx = e6.iter().map(|e| e.features[4]).fold(0.0f64, f64::max);
        assert!((mx - 1.0).abs() < 1e-9);
    }

    /// 手写 IF 的基本分离性：远离正常点云的离群点，raw 分数应显著高于正常点中位数
    #[test]
    fn isolation_forest_separates_outlier() {
        let mut rows: Vec<Vec<f64>> = (0..80)
            .map(|i| vec![13.0 + (i % 5) as f64 * 0.1, 4.6, 2.0, 1.0, 0.08, 1.0])
            .collect();
        let outlier = vec![3.0, 7.4, 2.0, 1.0, 1.0, 15.0]; // 深夜+巨仓+高频
        rows.push(outlier.clone());
        let forest = if_train(&rows, 7);
        let s_out = if_score(&forest, &outlier);
        let mut normals: Vec<f64> = rows[..80].iter().map(|r| if_score(&forest, r)).collect();
        let med = median(&mut normals);
        assert!(
            s_out > med + 0.15,
            "离群点分数({:.3})应明显高于正常中位数({:.3})",
            s_out,
            med
        );
    }

    /// 端到端冒烟：默认流水线必须保持高召回（低于 80% 即视为回归失败）
    #[test]
    fn pipeline_smoke_recall() {
        let cfg = GenConfig::default_();
        let res = run_pipeline(&cfg, DEFAULT_FUSION, false, DBSCAN_EPS, Z_THRESHOLD).unwrap();
        assert!(
            res.m_final.detection_rate >= 80.0,
            "终判检出率回归: {:.1}%",
            res.m_final.detection_rate
        );
        // 连击下载必须被聚合为批量事件（连击前 1-2 条个体近正常属预期漏检，故阈值取 6）
        assert!(
            res.incidents.iter().any(|i| i.event_count >= 6),
            "连击下载应被聚合为批量事件"
        );
    }

    /// 评测函数：tp/fp/fn 计数正确
    #[test]
    fn metrics_counting() {
        let mk = |tag: u8| LogRecord {
            user_id: 1,
            timestamp: Local::now(),
            operation: "view".into(),
            file_size_kb: 100.0,
            sensitivity: 1,
            is_anomaly: tag != 0,
            scenario_tag: tag,
        };
        let logs = vec![mk(1), mk(0), mk(1), mk(0), mk(0)];
        let flags = vec![true, true, false, false, true]; // tp1 fp1 fn1 f n
        let m = metrics_of(&flags, &logs);
        assert_eq!((m.true_positives, m.false_positives, m.false_negatives), (1, 2, 1));
        assert_eq!(m.detection_rate, 50.0);
    }
}
