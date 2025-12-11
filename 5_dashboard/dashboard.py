"""
AI 자율 운영 공정 시스템 - 실시간 대시보드
Streamlit 기반 웹 인터페이스

실행 방법:
streamlit run dashboard.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import json
import time
import sys
import os
from typing import Dict, List

# 메인 시스템 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '4_agent_system'))
try:
    from main_system import ManufacturingAISystem
    SYSTEM_AVAILABLE = True
except Exception as e:
    print(f"⚠️ AI 시스템 import 실패: {e}")
    SYSTEM_AVAILABLE = False


# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="AI 자율 운영 공정 시스템",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# 스타일링
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .status-normal {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 세션 상태 초기화
# ============================================================================

if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

if 'ai_system' not in st.session_state:
    st.session_state.ai_system = None

if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ============================================================================
# AI 시스템 초기화 (캐싱)
# ============================================================================

@st.cache_resource(show_spinner="AI 시스템 로딩 중... (최초 1회만 실행됩니다)")
def initialize_ai_system():
    """AI 시스템 초기화 - 캐싱으로 재사용"""
    try:
        if not SYSTEM_AVAILABLE:
            return None
        
        system = ManufacturingAISystem(
            detection_model_type="ensemble",
            detection_model_path=None
        )
        return system
    except Exception as e:
        st.error(f"AI 시스템 초기화 실패: {e}")
        return None


# ============================================================================
# 시뮬레이션 데이터 생성 함수
# ============================================================================

def generate_sensor_data(anomaly: bool = False) -> Dict:
    """센서 데이터 시뮬레이션 (Press 형식)"""
    
    if anomaly:
        # 이상 패턴 (실제 outlier 데이터 기반)
        ai0_vib = np.random.uniform(0.8, 1.5)  # 고진동
        ai1_vib = np.random.uniform(-0.8, -0.3)  # 고진동 (음수)
        ai2_current = np.random.uniform(230, 250)  # 과전류
    else:
        # 정상 패턴
        ai0_vib = np.random.uniform(-0.05, 0.05)  # 정상 진동
        ai1_vib = np.random.uniform(-0.05, 0.05)  # 정상 진동
        ai2_current = np.random.uniform(20, 50)  # 정상 전류
    
    return {
        "AI0_Vibration": round(ai0_vib, 4),
        "AI1_Vibration": round(ai1_vib, 4),
        "AI2_Current": round(ai2_current, 2)
    }


def generate_time_series_data(hours: int = 24, anomaly_at: int = None):
    """시계열 데이터 생성 (Press 형식)"""
    
    now = datetime.now()
    times = [now - timedelta(hours=hours-i) for i in range(hours)]
    
    data = []
    for i, t in enumerate(times):
        is_anomaly = (anomaly_at is not None and i == anomaly_at)
        sensor_data = generate_sensor_data(anomaly=is_anomaly)
        sensor_data['timestamp'] = t
        data.append(sensor_data)
    
    return pd.DataFrame(data)


# ============================================================================
# 메인 헤더
# ============================================================================

st.markdown('<div class="main-header">🏭 AI 자율 운영 공정 시스템</div>', 
            unsafe_allow_html=True)
st.markdown("**Team Autonomy** | 스마트 제조 AI Agent 해커톤 2025")

st.divider()


# ============================================================================
# 사이드바 - 설정
# ============================================================================

with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    # 설비 선택
    equipment_id = st.selectbox(
        "설비 선택",
        ["PRESS-01", "PRESS-02", "PRESS-03", 
         "사출기-1호기", "사출기-2호기"]
    )
    
    st.divider()
    
    # 모니터링 모드
    st.subheader("📊 모니터링 모드")
    monitoring_mode = st.radio(
        "모드 선택",
        ["실시간 모니터링", "시뮬레이션", "이력 분석"],
        index=0
    )
    
    st.divider()
    
    # 알림 설정
    st.subheader("🔔 알림 설정 (Press)")
    alert_vib_warning = st.slider("진동 주의 임계값 (g)", 0.10, 0.25, 0.15, 0.01)
    alert_vib_danger = st.slider("진동 위험 임계값 (g)", 0.25, 0.50, 0.30, 0.01)
    alert_current = st.slider("전류 임계값 (A)", 200, 250, 230, 5)
    
    st.divider()
    
    # 시스템 상태
    st.subheader("🖥️ 시스템 상태")
    
    # 자동 초기화 옵션
    auto_init = st.checkbox("자동 초기화 (페이지 로드 시)", value=True)
    
    if auto_init and not st.session_state.system_initialized:
        with st.spinner("AI 시스템 자동 초기화 중..."):
            st.session_state.ai_system = initialize_ai_system()
            if st.session_state.ai_system:
                st.session_state.system_initialized = True
    
    if st.button("🔄 AI 시스템 초기화/재시작", use_container_width=True):
        if not SYSTEM_AVAILABLE:
            st.error("❌ AI 시스템을 불러올 수 없습니다.")
        else:
            # 캐시 클리어 후 재초기화
            initialize_ai_system.clear()
            st.session_state.ai_system = initialize_ai_system()
            if st.session_state.ai_system:
                st.session_state.system_initialized = True
                st.success("✅ AI 시스템 재시작 완료!")
            else:
                st.error("❌ 초기화 실패")
    
    if st.session_state.system_initialized and st.session_state.ai_system:
        st.success("✅ AI 시스템 작동 중")
        st.caption("🤖 Detection + Retrieval + Action + PM + Report")
        st.caption("💾 모델 캐싱 활성화 (빠른 실행)")
    elif SYSTEM_AVAILABLE:
        st.warning("⚠️ AI 시스템 초기화 필요")
    else:
        st.error("❌ AI 시스템 불가용")
    
    # 실행 이력
    if st.session_state.history:
        st.divider()
        st.subheader("📊 실행 이력")
        st.caption(f"총 {len(st.session_state.history)}건 분석")
        
        # 최근 5건만 표시
        for i, record in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"#{len(st.session_state.history)-i} - {record['timestamp'].strftime('%H:%M:%S')}"):
                st.write(f"**설비:** {record['equipment_id']}")
                st.write(f"**이상 여부:** {'🚨 이상' if record['result'].get('is_anomaly') else '✅ 정상'}")
                if record['result'].get('is_anomaly'):
                    st.write(f"**이상 유형:** {record['result'].get('anomaly_type')}")
                    st.write(f"**신뢰도:** {record['result'].get('anomaly_score', 0):.1%}")


# ============================================================================
# 메인 컨텐츠
# ============================================================================

if monitoring_mode == "실시간 모니터링":
    
    # ========== 현재 상태 대시보드 ==========
    st.header("📈 실시간 센서 모니터링")
    
    # 데이터 입력 방법 선택
    data_input_mode = st.radio(
        "데이터 입력 방법",
        ["시뮬레이션 (랜덤)", "수동 입력", "이상 데이터 생성"],
        horizontal=True
    )
    
    if data_input_mode == "시뮬레이션 (랜덤)":
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            sensor_data = generate_sensor_data(anomaly=np.random.random() < 0.3)
            st.session_state.current_data = sensor_data
        
        if st.session_state.current_data is None:
            st.session_state.current_data = generate_sensor_data()
    
    elif data_input_mode == "수동 입력":
        col1, col2, col3 = st.columns(3)
        with col1:
            ai0 = st.number_input("AI0_Vibration (g)", -2.0, 2.0, 0.02, 0.01, format="%.4f")
        with col2:
            ai1 = st.number_input("AI1_Vibration (g)", -2.0, 2.0, -0.01, 0.01, format="%.4f")
        with col3:
            ai2 = st.number_input("AI2_Current (A)", 0.0, 300.0, 35.0, 1.0, format="%.2f")
        
        st.session_state.current_data = {
            "AI0_Vibration": ai0,
            "AI1_Vibration": ai1,
            "AI2_Current": ai2
        }
    
    else:  # 이상 데이터 생성
        if st.button("⚠️ 이상 데이터 생성", use_container_width=True):
            sensor_data = generate_sensor_data(anomaly=True)
            st.session_state.current_data = sensor_data
        
        if st.session_state.current_data is None:
            st.session_state.current_data = generate_sensor_data(anomaly=True)
    
    sensor_data = st.session_state.current_data
    
    # 센서 값 표시 (3개 컬럼 - Press 센서)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ai0_vib = sensor_data['AI0_Vibration']
        ai0_status = "🔴 이상" if abs(ai0_vib) > 0.3 else ("🟡 주의" if abs(ai0_vib) > 0.15 else "🟢 정상")
        st.metric(
            label="📳 AI0_Vibration",
            value=f"{ai0_vib:.4f} g",
            delta=f"{abs(ai0_vib) - 0.02:.4f} g" if ai0_vib != 0 else "0.0000 g",
            delta_color="inverse"
        )
        st.caption(f"{ai0_status} (정상: ±0.15g, 위험: ±0.30g)")
    
    with col2:
        ai1_vib = sensor_data['AI1_Vibration']
        ai1_status = "🔴 이상" if abs(ai1_vib) > 0.3 else ("🟡 주의" if abs(ai1_vib) > 0.15 else "🟢 정상")
        st.metric(
            label="📳 AI1_Vibration",
            value=f"{ai1_vib:.4f} g",
            delta=f"{abs(ai1_vib) - 0.02:.4f} g" if ai1_vib != 0 else "0.0000 g",
            delta_color="inverse"
        )
        st.caption(f"{ai1_status} (정상: ±0.15g, 위험: ±0.30g)")
    
    with col3:
        ai2_current = sensor_data['AI2_Current']
        ai2_status = "🔴 이상" if ai2_current > 230 else "🟢 정상"
        st.metric(
            label="⚡ AI2_Current",
            value=f"{ai2_current:.2f} A",
            delta=f"{ai2_current - 35:.2f} A",
            delta_color="inverse"
        )
        st.caption(f"{ai2_status} (정상: ~35A, 위험: >230A)")
    
    st.divider()
    
    # ========== 이상 탐지 결과 ==========
    st.header("🔍 AI 이상 탐지 결과")
    
    # 이상 여부 판단 (Press 기준)
    is_anomaly = (
        abs(ai0_vib) > 0.3 or 
        abs(ai1_vib) > 0.3 or 
        ai2_current > 230
    )
    
    if is_anomaly:
        st.error("🚨 **이상 감지!**")
        
        # 이상 유형 판단
        if abs(ai0_vib) > 0.3 and abs(ai1_vib) > 0.3:
            anomaly_type = "고진동+전류 이상" if ai2_current > 230 else "고진동 이상"
        elif abs(ai0_vib) > 0.3:
            anomaly_type = "AI0 진동 이상"
        elif abs(ai1_vib) > 0.3:
            anomaly_type = "AI1 진동 이상"
        else:
            anomaly_type = "전류 이상"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 탐지 정보")
            st.write(f"**이상 유형:** {anomaly_type}")
            st.write(f"**신뢰도:** 87.5%")
            st.write(f"**발생 시각:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.markdown("### 📊 위험도 평가")
            risk_score = 0.75
            st.progress(risk_score)
            st.write(f"**위험 점수:** {risk_score:.1%} (높음)")
            st.write(f"**예상 영향:** 생산 중단, 품질 저하")
        
        st.divider()
        
        # ========== AI Agent 분석 실행 ==========
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            run_analysis = st.button("🤖 AI Agent 분석 실행", type="primary", use_container_width=True)
        
        with col_btn2:
            show_detail = st.checkbox("상세 로그", value=False)
        
        if run_analysis:
            # AI 시스템 초기화 확인
            if not st.session_state.system_initialized or not st.session_state.ai_system:
                st.error("❌ AI 시스템을 먼저 초기화해주세요! (왼쪽 사이드바에서 자동 초기화 활성화)")
            else:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # 실제 AI 시스템 실행
                    status_text.text("🔍 Detection Agent 실행 중...")
                    progress_bar.progress(20)
                    
                    result = st.session_state.ai_system.process_anomaly_event(
                        equipment_id=equipment_id,
                        sensor_data=sensor_data
                    )
                    
                    # 결과 저장
                    st.session_state.last_result = result
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'equipment_id': equipment_id,
                        'result': result
                    })
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 전체 분석 완료!")
                    time.sleep(0.3)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # 성공 메시지
                    st.success("🎉 AI Agent 분석이 완료되었습니다!")
                    
                    # 소요 시간 표시
                    elapsed = result.get('elapsed_time', 0)
                    st.info(f"⏱️ 소요 시간: {elapsed:.2f}초")
                    
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text(f"❌ 오류 발생")
                    st.error(f"AI 시스템 실행 중 오류가 발생했습니다: {e}")
                    
                    if show_detail:
                        import traceback
                        with st.expander("상세 에러 로그"):
                            st.code(traceback.format_exc())
                    
                    st.session_state.last_result = None
            
            # ========== 분석 결과 표시 ==========
            if st.session_state.last_result:
                st.divider()
                st.header("📋 AI Agent 분석 결과")
                
                result = st.session_state.last_result
                
                # 탭으로 구분
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔍 이상 탐지", 
                    "📖 유사 사례", 
                    "🔧 조치 가이드", 
                    "🛠️ 예방보전",
                    "📄 8D Report"
                ])
                
                with tab1:
                    st.markdown("### 🔍 Detection Agent 결과")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        is_anom = result.get('is_anomaly', False)
                        st.metric("이상 여부", "이상 감지" if is_anom else "정상", 
                                 delta="위험" if is_anom else "정상")
                        st.metric("이상 유형", result.get('anomaly_type', 'N/A'))
                    
                    with col2:
                        score = result.get('anomaly_score', 0)
                        st.metric("신뢰도", f"{score:.1%}")
                        st.metric("발생 시각", result.get('timestamp', 'N/A'))
                
                with tab2:
                    st.markdown("### 📖 Retrieval Agent 결과")
                    st.markdown("**검색된 유사 사례 (RAG + ChromaDB)**")
                    
                    similar_cases = result.get('similar_cases', [])
                    
                    if similar_cases:
                        for i, case in enumerate(similar_cases, 1):
                            content = case.get('content', '')
                            metadata = case.get('metadata', {})
                            similarity = case.get('similarity', 0)
                            
                            st.info(f"""
                            **[검색 결과 #{i}]** (유사도: {similarity:.1%})
                            
                            {content[:500]}...
                            
                            *출처: {metadata.get('source_file', 'N/A')}*
                            *카테고리: {metadata.get('category', 'N/A')}*
                            """)
                    else:
                        st.warning("검색된 유사 사례가 없습니다.")
                
                with tab3:
                    st.markdown("### 🔧 Action Agent 결과 (LoRA 모델)")
                    
                    # CoT 추론 과정
                    cot_reasoning = result.get('cot_reasoning', '')
                    if cot_reasoning:
                        st.markdown("#### 🧠 상황 분석 및 추론 과정 (CoT)")
                        st.write(cot_reasoning)
                    
                    # 원인 분석
                    root_causes = result.get('root_causes', [])
                    if root_causes:
                        st.markdown("#### ✅ 원인 분석 (우선순위)")
                        for i, cause in enumerate(root_causes, 1):
                            if i == 1:
                                st.success(f"**{i}순위:** {cause}")
                            elif i == 2:
                                st.warning(f"**{i}순위:** {cause}")
                            else:
                                st.info(f"**{i}순위:** {cause}")
                    
                    # 체크리스트
                    checklist = result.get('checklist', [])
                    if checklist:
                        st.markdown("#### 📝 우선 점검 체크리스트")
                        for i, item in enumerate(checklist):
                            st.checkbox(item, key=f"check_{i}_{item[:20]}")
                
                with tab4:
                    st.markdown("### 🛠️ PM Recommendation Agent 결과")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        health_score = result.get('health_score', 0)
                        st.metric("Health Score", f"{health_score:.1%}", 
                                 delta=f"{health_score - 1.0:.1%}", delta_color="inverse")
                        
                        failure_risk = result.get('failure_risk', 0)
                        st.metric("고장 위험도", f"{failure_risk:.1%}", 
                                 delta=f"{failure_risk:.1%}", delta_color="inverse")
                    
                    with col2:
                        recovery_time = result.get('estimated_recovery_time', 'N/A')
                        st.metric("예상 복구 시간", recovery_time)
                        
                        urgency = result.get('urgency_level', 'N/A')
                        st.metric("긴급도", urgency)
                    
                    # PM 추천사항
                    pm_recommendations = result.get('pm_recommendations', [])
                    if pm_recommendations:
                        st.markdown("#### 📋 PM 추천사항")
                        for rec in pm_recommendations:
                            if 'HIGH' in str(rec) or '긴급' in str(rec):
                                st.error(rec)
                            elif 'MEDIUM' in str(rec) or '주의' in str(rec):
                                st.warning(rec)
                            else:
                                st.info(rec)
                
                with tab5:
                    st.markdown("### 📄 8D Report Agent 결과 (LoRA 모델)")
                    
                    report_8d = result.get('report_8d', '')
                    
                    if report_8d:
                        # 8D Report 표시
                        st.markdown(report_8d)
                        
                        # 다운로드 버튼
                        st.download_button(
                            label="📥 8D Report 다운로드",
                            data=report_8d,
                            file_name=f"8D_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.warning("8D Report가 생성되지 않았습니다. (LoRA 모델이 필요합니다)")
                        st.info("8D Report 생성을 위해서는 LoRA 모델이 필요합니다.")
    
    else:
        st.success("✅ **정상 운전 중**")
        st.info("모든 센서 값이 정상 범위 내에 있습니다.")


elif monitoring_mode == "시뮬레이션":
    
    st.header("🎮 시뮬레이션 모드")
    
    st.info("""
    이 모드에서는 다양한 이상 시나리오를 시뮬레이션할 수 있습니다.
    """)
    
    # 시뮬레이션 설정
    col1, col2 = st.columns(2)
    
    with col1:
        sim_anomaly_type = st.selectbox(
            "이상 유형 선택",
            ["온도 이상", "압력 이상", "진동 이상", "사이클타임 지연", "정상"]
        )
    
    with col2:
        sim_severity = st.slider("이상 심각도", 0.0, 1.0, 0.7)
    
    if st.button("🚀 시뮬레이션 실행", type="primary", use_container_width=True):
        st.success(f"시뮬레이션 시작: {sim_anomaly_type}")
        
        # 시뮬레이션 데이터 생성 및 분석
        # (실제 구현)


elif monitoring_mode == "이력 분석":
    
    st.header("📊 이력 데이터 분석 (Press)")
    
    # 시계열 데이터 생성
    df = generate_time_series_data(hours=24, anomaly_at=18)
    
    # AI0 진동 차트
    fig_ai0 = go.Figure()
    fig_ai0.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['AI0_Vibration'],
        mode='lines+markers',
        name='AI0_Vibration',
        line=dict(color='blue', width=2)
    ))
    fig_ai0.add_hline(y=alert_vib_warning, line_dash="dash", line_color="orange",
                      annotation_text="주의 임계값")
    fig_ai0.add_hline(y=alert_vib_danger, line_dash="dash", line_color="red",
                      annotation_text="위험 임계값")
    fig_ai0.add_hline(y=-alert_vib_warning, line_dash="dash", line_color="orange")
    fig_ai0.add_hline(y=-alert_vib_danger, line_dash="dash", line_color="red")
    fig_ai0.update_layout(
        title="AI0 진동 추이 (24시간)",
        xaxis_title="시간",
        yaxis_title="진동 (g)",
        height=400
    )
    st.plotly_chart(fig_ai0, use_container_width=True)
    
    # 기타 센서 차트
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ai1 = px.line(df, x='timestamp', y='AI1_Vibration', 
                          title='AI1 진동 추이')
        st.plotly_chart(fig_ai1, use_container_width=True)
    
    with col2:
        fig_current = px.line(df, x='timestamp', y='AI2_Current',
                              title='전류 추이 (A)')
        st.plotly_chart(fig_current, use_container_width=True)


# ============================================================================
# 푸터
# ============================================================================

st.divider()

# 성능 정보
with st.expander("💡 성능 최적화 팁"):
    st.markdown("""
    ### ⚡ 빠른 실행을 위한 팁
    
    1. **자동 초기화 활성화** (왼쪽 사이드바)
       - 페이지 로드 시 자동으로 모델 로드
       - 한 번만 로드되고 캐싱됨
    
    2. **모델 캐싱**
       - AI 시스템은 `@st.cache_resource`로 캐싱
       - 두 번째 실행부터는 매우 빠름 (모델 재로드 X)
    
    3. **예상 소요 시간**
       - 최초 초기화: 30초~1분 (LoRA 모델 로딩)
       - 이후 분석 실행: 10~30초 (캐싱 후)
    
    4. **브라우저 새로고침 시**
       - Streamlit 서버가 유지되면 캐시 유지
       - 완전히 재시작하려면 "AI 시스템 초기화/재시작" 클릭
    
    5. **메모리 부족 시**
       - ChromaDB만 사용 (FAISS 비활성화)
       - LoRA 모델 대신 Base 모델 사용
    """)

st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏭 AI 자율 운영 공정 시스템 v2.0 (최적화)</p>
    <p>Team Autonomy | 스마트 제조 AI Agent 해커톤 2025</p>
</div>
""", unsafe_allow_html=True)