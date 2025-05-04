"""
열거형 정의 모듈
시스템에서 사용되는 상수들을 정의
"""
from enum import Enum


class MemoryTier(Enum):
    """메모리 계층 열거형"""
    WORKING = "working"      # Redis - 워킹 메모리
    SHORT_TERM = "short"     # SQLite - 단기 메모리 
    LONG_TERM = "long"       # ChromaDB - 장기 메모리


class ConceptType(Enum):
    """개념 타입 열거형"""
    PERSON = "person"        # 인물
    OBJECT = "object"        # 객체
    EVENT = "event"          # 이벤트
    TIME = "time"            # 시간
    PLACE = "place"          # 장소
    ABSTRACT = "abstract"    # 추상적 개념


class ConnectionType(Enum):
    """연관 타입 열거형"""
    SEMANTIC = "semantic"    # 의미적 관계
    TEMPORAL = "temporal"    # 시간적 관계
    SPATIAL = "spatial"      # 공간적 관계
    CAUSAL = "causal"        # 인과 관계
    EMOTIONAL = "emotional"  # 감정적 관계
    PROCEDURAL = "procedural" # 절차적 관계

