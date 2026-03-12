import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
	# ==========================================
	# 1. DTW 核心算法
	# ==========================================
def calculate_dtw_distance(s1, s2):
	    n, m = len(s1), len(s2)
	    dtw_matrix = np.full((n + 1, m + 1), np.inf)
	    dtw_matrix[0, 0] = 0
	    for i in range(1, n + 1):
	        for j in range(1, m + 1):
	            cost = (s1[i - 1] - s2[j - 1]) ** 2
	            dtw_matrix[i, j] = cost + min(
	                dtw_matrix[i - 1, j],    # 插入
	                dtw_matrix[i, j - 1],    # 删除
	                dtw_matrix[i - 1, j - 1] # 匹配
	            )
	    # 归一化距离
	    return np.sqrt(dtw_matrix[n, m]) / ((n + m) / 2)
def normalize_series(series):
	    if np.std(series) == 0:
	        return series - np.mean(series)
	    return (series - np.mean(series)) / np.std(series)
	# ==========================================
	# 2. 数据读取与模拟生成 
	# ==========================================
def load_and_process_data(file_path):
	    print(f"正在读取文件: {file_path} ...")
	    if not os.path.exists(file_path):
	        print(">> 文件不存在，启动【故障模拟模式】...")
	        return generate_simulation_data()
	    # 真实Excel读取逻辑
	    try:
	        df = pd.read_excel(file_path)
	        # 简单的自动列名匹配
	        t0_col = [c for c in df.columns if 't0' in c.lower() or 'start' in c.lower()]
	        t1_col = [c for c in df.columns if 't1' in c.lower() or 'end' in c.lower()]
	        if t0_col and t1_col:
	            df['P1'] = df[t1_col[0]] - df[t0_col[0]]
	            df = df[df['P1'] > 0].reset_index(drop=True)
	            df['cycle'] = df.index
	            return df
	        else:
	            print(">> 未识别列名，启用模拟数据")
	            return generate_simulation_data()
	    except Exception as e:
	        print(f">> 读取失败: {e}，启用模拟数据")
	        return generate_simulation_data()
def generate_simulation_data():
	    """
	    生成一个完整的故障生命周期数据：
	    阶段1: 健康运行 (0-150周期)
	    阶段2: 缓慢漏气/劣化 (150-300周期) -> 这里的报警是预测性维护的关键
	    阶段3: 严重卡滞/故障 (300-400周期)
	    """
	    np.random.seed(42)
	    # --- 阶段1: 健康基准 (150个周期) ---
	    # P1 稳定在 0.50s 左右，微小随机波动
	    p1_healthy = np.random.normal(0.50, 0.02, 150)
	    # --- 阶段2: 缓慢劣化 (150个周期) ---
	    # 模拟气缸开始漏气，P1 从 0.50 缓慢爬升到 0.90
	    # 这种爬升趋势是DTW最擅长捕捉的
	    degradation_trend = np.linspace(0.50, 0.90, 150)
	    # 加入一点随机扰动，让它看起来像真实数据
	    p1_degrading = degradation_trend + np.random.normal(0, 0.03, 150)
	    # --- 阶段3: 严重故障 (100个周期) ---
	    # 气缸动作严重迟缓，P1 维持在高位 (1.2s)，且波动剧烈
	    p1_fault = np.random.normal(1.20, 0.10, 100)
	    # 合并数据
	    p1_all = np.concatenate([p1_healthy, p1_degrading, p1_fault])
	    # 构造 DataFrame
	    df = pd.DataFrame({
	        'cycle': range(len(p1_all)),
	        't0': range(len(p1_all)),         # 模拟时间戳
	        't1': range(len(p1_all)) + p1_all,
	        'P1': p1_all
	    })
	    print(f">> 模拟数据生成完毕: 共 {len(df)} 个周期")
	    print(f">> 数据包含: 健康期(0-150) -> 劣化期(150-300) -> 故障期(300-400)")
	    return df
	# ==========================================
	# 3. 监测逻辑
	# ==========================================
def run_monitoring_process(df, K=100, M=50, threshold=0.15):
	    p1_series = df['P1'].values
	    if len(p1_series) < K:
	        print("数据过短"); return None, None, None, None
	    # 1. 提取基准指纹 (取前K个点，默认为健康)
	    s_ref_raw = p1_series[:K]
	    s_ref = normalize_series(s_ref_raw)
	    dtw_distances = []
	    is_alert = []
	    # 2. 滑动窗口监测
	    for i in range(M, len(p1_series)):
	        s_real_raw = p1_series[i-M : i]
	        s_real = normalize_series(s_real_raw)
	        dist = calculate_dtw_distance(s_ref, s_real)
	        dtw_distances.append(dist)
	        # 记录报警状态
	        if dist > threshold:
	            is_alert.append(1)
	        else:
	            is_alert.append(0)
	    return s_ref_raw, dtw_distances, is_alert, threshold
	# ==========================================
	# 4. 可视化分析 (增强版)
	# ==========================================
def visualize_results(df, s_ref, dtw_distances, is_alert, threshold, M=1):
	    plt.figure(figsize=(14, 12)) # 调大画布
	    # --- 子图1: P1 原始趋势与故障阶段 ---
	    ax1 = plt.subplot(3, 1, 1)
	    ax1.plot(df['cycle'], df['P1'], label='P1 Value (Grip Duration)', color='steelblue', linewidth=1.5)
	    # 标记基准区域
	    ax1.axvspan(0, len(s_ref), color='green', alpha=0.15, label='Health Reference (Train)')
	    # 标记模拟故障阶段背景 (仅作示意)
	    #ax1.axvspan(150, 300, color='yellow', alpha=0.1, label='Degradation Stage')
	    #ax1.axvspan(300, 400, color='red', alpha=0.1, label='Fault Stage')
	    ax1.set_title('Cylinder P1 Feature Lifecycle', fontsize=12, fontweight='bold')
	    ax1.set_ylabel('Time (s)')
	    ax1.legend(loc='upper left')
	    ax1.grid(True, linestyle='--', alpha=0.4)
	    # --- 子图2: DTW 距离与报警点 ---
	    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
	    x_axis_dtw = range(M, len(df))
	    dtw_arr = np.array(dtw_distances)
	    ax2.plot(x_axis_dtw, dtw_arr, label='DTW Distance', color='orange', linewidth=1.5)
	    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Alarm Threshold ({threshold})')
	    # 填充报警区域
	    ax2.fill_between(x_axis_dtw, 0, dtw_arr, where=dtw_arr>threshold, 
	                     color='red', alpha=0.3, label='Alert Zone')
	    #标记首次报警点
	    alert_indices = np.where(np.array(is_alert) == 1)[0]
	    if len(alert_indices) > 0:
	        first_alert_idx = alert_indices[0]
	        # 映射回真实周期
	        first_alert_cycle = first_alert_idx + M
	        ax2.scatter([first_alert_cycle], [dtw_distances[first_alert_idx]], 
	                    color='red', s=100, zorder=5, edgecolors='black', label='First Alert')
	        ax2.annotate(f'First Alert\nCycle {first_alert_cycle}', 
	                     xy=(first_alert_cycle, dtw_distances[first_alert_idx]),
	                     xytext=(first_alert_cycle+20, dtw_distances[first_alert_idx]+0.2),
	                     arrowprops=dict(facecolor='black', arrowstyle='->'),
	                     fontweight='bold')
	    ax2.set_title('DTW Anomaly Score (Early Warning Indicator)', fontsize=12, fontweight='bold')
	    ax2.set_ylabel('Distance')
	    ax2.legend(loc='upper left')
	    ax2.grid(True, linestyle='--', alpha=0.4)
	    # --- 子图3: 波形形态对比 ---
	    ax3 = plt.subplot(3, 1, 3)
	    # 绘制健康基准
	    ax3.plot(normalize_series(s_ref), label='Reference (Healthy)', color='green', linewidth=2.5, alpha=0.8)
	    # 绘制故障时刻的波形
	    if len(alert_indices) > 0:
	        # 取报警最强烈的时候
	        max_alert_idx = np.argmax(dtw_distances)
	        end_idx = M + max_alert_idx
	        start_idx = end_idx - M
	        s_anomaly = normalize_series(df['P1'].values[start_idx : end_idx])
	        ax3.plot(s_anomaly, label='Current (Anomaly)', color='red', linewidth=2.5, linestyle='--')
	        ax3.set_title(f'Shape Comparison at Peak Anomaly (Cycle {end_idx})', fontsize=12, fontweight='bold')
	    ax3.set_xlabel('Time Steps in Window')
	    ax3.legend()
	    ax3.grid(True, linestyle='--', alpha=0.4)
	    plt.tight_layout()
	    plt.show()
	# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
	    # 1. 获取数据 (这里会自动生成模拟数据)
        df_data = load_and_process_data(r"D:\Hjiang\work doc\standard cylinder\cylinder.xlsx") 
        output_filename = r"D:\Hjiang\work doc\standard cylinder\TS_switch\simulation_output.xlsx"
        try:
            df_data.to_excel(output_filename, index=False)
            print(f">> 模拟数据已成功保存至文件: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f">> 保存文件时出错: {e}")
        # 2. 设置参数
	    # 阈值设为 0.5，以便在劣化阶段(150-300)触发报警，而不是等到故障阶段(300+)
        config = {
	        "K": 100,        # 基准长度
	        "M":50 ,        # 滑动窗口
	        "THRESHOLD": 0.15# 报警阈值
	    }
	    # 3. 运行监测
        s_ref, distances, alerts, thresh = run_monitoring_process(
	        df_data, 
	        K=config["K"], 
	        M=config["M"], 
	        threshold=config["THRESHOLD"]
	    )
	    # 4. 结果输出与可视化
if distances:
	        total_alerts = sum(alerts)
	        print(f"\n========== 监测报告 ==========")
	        print(f"总周期数: {len(df_data)}")
	        print(f"报警次数: {total_alerts}")
	        if total_alerts > 0:
	            # 找到第一次报警的周期
	            first_alert_cycle = np.where(np.array(alerts)==1)[0][0] + config["M"]
	            print(f"首次预警时间: 第 {first_alert_cycle} 周期")
	            print(f"结论: 设备在劣化阶段即被成功捕获，建议在此时进行维护。")
	        else:
	            print("结论: 设备运行正常。")
	        visualize_results(df_data, s_ref, distances, alerts, thresh, config["M"])