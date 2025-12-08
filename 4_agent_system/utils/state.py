"""
상태 정의
"""
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any
import operator

class AgentState(TypedDict):
    """Multi-Agent 시스템 전체 상태"""
    # 입력
    equipment_id: str
    timestamp: str
    sensor_data: Dict[str, Any]
    
    # Detection Agent 출력
    is_anomaly: bool
    anomaly_type: str
    anomaly_score: float
    
    # Retrieval Agent 출력
    rag_context: str
    similar_cases: List[Dict]
    
    # Action Agent 출력
    root_causes: List[Dict]
    action_guide: str
    checklist: List[str]
    
    # PM Agent 출력
    health_score: float
    failure_risk: float
    pm_recommendations: List[Dict]
    
    # Report Agent 출력
    report_8d: str
    
    # 메시지 로그
    messages: Annotated[List[str], operator.add]
    
    # 메타데이터
    workflow_start_time: str
    workflow_status: str