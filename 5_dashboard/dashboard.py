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
from typing import Dict, List

# 메인 시스템 import (실제 파일명에 맞게 수정)
# from main_system import ManufacturingAISystem


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


# ============================================================================
# 시뮬레이션 데이터 생성 함수
# ============================================================================

def generate_sensor_data(anomaly: bool = False) -> Dict:
    """센서 데이터 시뮬레이션"""
    
    if anomaly:
        # 이상 패턴
        anomaly_type = np.random.choice(['온도', '압력', '진동', '사이클타임'])
        
        if anomaly_type == '온도':
            temp = np.random.uniform(230, 250)
            pressure = np.random.uniform(110, 130)
            vibration = np.random.uniform(0.8, 1.5)
            cycle_time = np.random.uniform(48, 58)
        elif anomaly_type == '압력':
            temp = np.random.uniform(190, 210)
            pressure = np.random.uniform(70, 90)
            vibration = np.random.uniform(0.8, 1.5)
            cycle_time = np.random.uniform(58, 72)
        elif anomaly_type == '진동':
            temp = np.random.uniform(190, 210)
            pressure = np.random.uniform(110, 130)
            vibration = np.random.uniform(3.0, 5.0)
            cycle_time = np.random.uniform(48, 58)
        else:  # 사이클타임
            temp = np.random.uniform(190, 210)
            pressure = np.random.uniform(110, 130)
            vibration = np.random.uniform(0.8, 1.5)
            cycle_time = np.random.uniform(75, 90)
    else:
        # 정상 패턴
        temp = np.random.uniform(195, 205)
        pressure = np.random.uniform(115, 125)
        vibration = np.random.uniform(0.8, 1.5)
        cycle_time = np.random.uniform(48, 52)
    
    return {
        "temperature": round(temp, 1),
        "pressure": round(pressure, 1),
        "vibration": round(vibration, 2),
        "cycle_time": round(cycle_time, 1)
    }


def generate_time_series_data(hours: int = 24, anomaly_at: int = None):
    """시계열 데이터 생성"""
    
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
        ["사출기-1호기", "사출기-2호기", "사출기-3호기", 
         "프레스-1호기", "CNC-1호기"]
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
    st.subheader("🔔 알림 설정")
    alert_temp = st.slider("온도 임계값 (°C)", 210, 240, 225)
    alert_pressure = st.slider("압력 임계값 (bar)", 90, 110, 100)
    alert_vibration = st.slider("진동 임계값 (mm/s)", 2.0, 3.5, 2.5)
    
    st.divider()
    
    # 시스템 상태
    st.subheader("🖥️ 시스템 상태")
    
    if st.button("🔄 시스템 초기화", use_container_width=True):
        with st.spinner("시스템 초기화 중..."):
            time.sleep(2)
            st.session_state.system_initialized = True
            st.success("✅ 초기화 완료!")
    
    if st.session_state.system_initialized:
        st.success("✅ 시스템 작동 중")
    else:
        st.warning("⚠️ 시스템 초기화 필요")


# ============================================================================
# 메인 컨텐츠
# ============================================================================

if monitoring_mode == "실시간 모니터링":
    
    # ========== 현재 상태 대시보드 ==========
    st.header("📈 실시간 센서 모니터링")
    
    # 센서 데이터 생성 (시뮬레이션)
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        sensor_data = generate_sensor_data(anomaly=np.random.random() < 0.3)
        st.session_state.current_data = sensor_data
    
    if st.session_state.current_data is None:
        st.session_state.current_data = generate_sensor_data()
    
    sensor_data = st.session_state.current_data
    
    # 센서 값 표시 (4개 컬럼)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = sensor_data['temperature']
        temp_status = "🔴 이상" if temp > alert_temp else "🟢 정상"
        st.metric(
            label="🌡️ 온도",
            value=f"{temp}°C",
            delta=f"{temp - 200:.1f}°C",
            delta_color="inverse"
        )
        st.caption(temp_status)
    
    with col2:
        pressure = sensor_data['pressure']
        pressure_status = "🔴 이상" if pressure < alert_pressure else "🟢 정상"
        st.metric(
            label="💨 압력",
            value=f"{pressure} bar",
            delta=f"{pressure - 120:.1f} bar",
            delta_color="inverse"
        )
        st.caption(pressure_status)
    
    with col3:
        vibration = sensor_data['vibration']
        vib_status = "🔴 이상" if vibration > alert_vibration else "🟢 정상"
        st.metric(
            label="📳 진동",
            value=f"{vibration} mm/s",
            delta=f"{vibration - 1.0:.2f} mm/s",
            delta_color="inverse"
        )
        st.caption(vib_status)
    
    with col4:
        cycle_time = sensor_data['cycle_time']
        cycle_status = "🔴 이상" if cycle_time > 65 else "🟢 정상"
        st.metric(
            label="⏱️ 사이클 타임",
            value=f"{cycle_time} 초",
            delta=f"{cycle_time - 50:.1f} 초",
            delta_color="inverse"
        )
        st.caption(cycle_status)
    
    st.divider()
    
    # ========== 이상 탐지 결과 ==========
    st.header("🔍 AI 이상 탐지 결과")
    
    # 이상 여부 판단
    is_anomaly = (
        temp > alert_temp or 
        pressure < alert_pressure or 
        vibration > alert_vibration or
        cycle_time > 65
    )
    
    if is_anomaly:
        st.error("🚨 **이상 감지!**")
        
        # 이상 유형 판단
        if temp > alert_temp:
            anomaly_type = "온도 이상"
        elif pressure < alert_pressure:
            anomaly_type = "압력 이상"
        elif vibration > alert_vibration:
            anomaly_type = "진동 이상"
        else:
            anomaly_type = "사이클타임 지연"
        
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
        if st.button("🤖 AI Agent 분석 실행", type="primary", use_container_width=True):
            
            with st.spinner("AI Agent가 분석 중입니다..."):
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Detection
                status_text.text("🔍 Detection Agent: 이상 탐지 중...")
                time.sleep(0.5)
                progress_bar.progress(20)
                
                # Retrieval
                status_text.text("📖 Retrieval Agent: 유사 사례 검색 중...")
                time.sleep(1.0)
                progress_bar.progress(40)
                
                # Action
                status_text.text("🔧 Action Agent: 조치 가이드 생성 중... (LoRA)")
                time.sleep(1.5)
                progress_bar.progress(60)
                
                # PM
                status_text.text("🛠️ PM Agent: 예방보전 분석 중...")
                time.sleep(0.8)
                progress_bar.progress(80)
                
                # Report
                status_text.text("📄 Report Agent: 8D Report 생성 중... (LoRA)")
                time.sleep(1.2)
                progress_bar.progress(100)
                
                status_text.text("✅ 분석 완료!")
                time.sleep(0.5)
            
            st.success("🎉 AI Agent 분석이 완료되었습니다!")
            
            # ========== 분석 결과 표시 ==========
            st.divider()
            st.header("📋 AI Agent 분석 결과")
            
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
                    st.metric("이상 여부", "이상 감지", delta="위험")
                    st.metric("이상 유형", anomaly_type)
                
                with col2:
                    st.metric("신뢰도", "87.5%")
                    st.metric("발생 시각", datetime.now().strftime("%H:%M:%S"))
            
            with tab2:
                st.markdown("### 📖 Retrieval Agent 결과")
                st.markdown("**검색된 유사 사례 (RAG)**")
                
                st.info("""
                **[과거 이력 #2023-08-15]** (유사도: 92%)
                - 설비: 사출기-2호기
                - 증상: 실린더 온도 급상승 (235°C)
                - 원인: 히터 코일 단선
                - 조치: 히터 교체 후 정상화
                - 소요시간: 4시간
                """)
                
                st.info("""
                **[설비 매뉴얼 3.2절]** (유사도: 88%)
                - 실린더 온도가 설정값 ±15°C를 벗어날 경우
                - 히터 저항값 측정 (정상: 30~35Ω)
                - 열전대 센서 점검 필요
                """)
            
            with tab3:
                st.markdown("### 🔧 Action Agent 결과 (LoRA 모델)")
                
                st.markdown("#### 🧠 상황 분석 및 추론 과정 (CoT)")
                st.write("""
                **1단계: 데이터 이상 징후 확인**
                - 실린더 온도 235°C (정상 200°C 대비 +35°C, 17.5% 변동)
                - 설정 임계값(±15°C)을 명확히 초과
                - 패턴: 전형적인 **온도 이상** 징후
                
                **2단계: 근거 자료 교차 검증**
                - RAG 시스템 검색 결과와 **92% 일치**
                - 과거 이력에서 동일한 센서 패턴 확인
                
                **3단계: 물리적 인과관계 분석**
                - 예상 현상: 열전달 효율 저하
                - 히터 고장의 전형적인 증상과 일치
                
                **4단계: 최종 결론**
                → 근본 원인: **히터 고장 또는 성능 저하**
                → 확률: **높음 (85% 이상)**
                """)
                
                st.markdown("#### ✅ 원인 분석 (우선순위)")
                st.success("**1순위: 히터 고장 또는 성능 저하** (확률 85%)")
                st.warning("**2순위: 온도 센서 오류** (확률 30%)")
                st.info("**3순위: 냉각 시스템 막힘** (확률 15%)")
                
                st.markdown("#### 📝 우선 점검 체크리스트")
                checklist = [
                    "경보 이력 및 트렌드 데이터 확인",
                    "육안 점검 (누유, 균열, 변색)",
                    "히터 저항값 측정 (정상: 30~35Ω)",
                    "열전대 센서 점검",
                    "온도 제어기 파라미터 확인"
                ]
                for item in checklist:
                    st.checkbox(item, key=f"check_{item}")
            
            with tab4:
                st.markdown("### 🛠️ PM Recommendation Agent 결과")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Health Score", "55%", delta="-30%", delta_color="inverse")
                    st.metric("고장 위험도", "65%", delta="+40%", delta_color="inverse")
                
                with col2:
                    st.metric("예상 복구 시간", "4~6시간")
                    st.metric("권장 조치", "48시간 내 긴급 점검")
                
                st.markdown("#### 📋 PM 추천사항")
                st.error("""
                **[HIGH] 48시간 내 긴급 점검 필요**
                - 히터 교체 검토
                - 온도 제어 시스템 전면 점검
                - 전문가 진단 요청
                - 예상 소요 시간: 4~6시간
                """)
            
            with tab5:
                st.markdown("### 📄 8D Report Agent 결과 (LoRA 모델)")
                
                report = f"""
**D1. 팀 구성**
- 대상 설비: {equipment_id}
- 담당 부서: 생산기술팀, 품질팀, 설비보전팀
- 발생 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**D2. 문제 정의**
- 현상: {anomaly_type} 발생으로 정상 가동 불가
- 영향 범위: 생산 중단, 품질 이슈 발생 가능
- 긴급도: 높음

**D3. 임시 조치 (ICA)**
- 설비 즉시 정지 및 안전 조치 완료
- 생산 중 제품 격리 및 검사 대기
- 대체 설비로 생산 전환

**D4. 근본 원인 분석 (RCA)**
- 추정 원인: 히터 고장 또는 성능 저하
- 분석 근거: 센서 데이터 분석, RAG 과거 이력 검토
- 확률: 85% 이상

**D5. 영구 대책 (PCA)**
- 히터 교체 및 예비품 확보
- 예방보전(PM) 주기 재설정
- 온도 모니터링 시스템 강화

**D6. 대책 실행 및 검증**
- 조치 완료 후 48시간 연속 모니터링
- 성능 테스트 및 품질 검증

**D7. 재발 방지**
- 정기 점검 항목에 히터 저항값 측정 추가
- 작업 표준서(SOP) 개정
- 전 직원 교육 실시
                """
                
                st.code(report, language="markdown")
                
                st.download_button(
                    label="📥 8D Report 다운로드",
                    data=report,
                    file_name=f"8D_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
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
    
    st.header("📊 이력 데이터 분석")
    
    # 시계열 데이터 생성
    df = generate_time_series_data(hours=24, anomaly_at=18)
    
    # 온도 차트
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines+markers',
        name='온도',
        line=dict(color='red', width=2)
    ))
    fig_temp.add_hline(y=alert_temp, line_dash="dash", line_color="orange",
                       annotation_text="임계값")
    fig_temp.update_layout(
        title="온도 추이 (24시간)",
        xaxis_title="시간",
        yaxis_title="온도 (°C)",
        height=400
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # 기타 센서 차트
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pressure = px.line(df, x='timestamp', y='pressure', 
                               title='압력 추이')
        st.plotly_chart(fig_pressure, use_container_width=True)
    
    with col2:
        fig_vibration = px.line(df, x='timestamp', y='vibration',
                                title='진동 추이')
        st.plotly_chart(fig_vibration, use_container_width=True)


# ============================================================================
# 푸터
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏭 AI 자율 운영 공정 시스템 v2.0</p>
    <p>Team Autonomy | 스마트 제조 AI Agent 해커톤 2025</p>
</div>
""", unsafe_allow_html=True)