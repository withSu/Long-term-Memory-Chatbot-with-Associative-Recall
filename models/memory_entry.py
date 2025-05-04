"""
메모리 엔트리 데이터 모델
메모리의 기본 데이터 구조를 정의
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from models.enums import MemoryTier


@dataclass
class MemoryEntry:
    """
    메모리 엔트리 클래스
    모든 메모리 데이터의 기본 단위
    """
    id: str
    content: Dict[str, Any]
    concepts: List[str]
    importance: float = 0.5
    emotional_weight: float = 0.0
    access_count: int = 0
    tier: MemoryTier = MemoryTier.WORKING
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None
    creation_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """메모리 엔트리를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'content': self.content,
            'concepts': self.concepts,
            'importance': self.importance,
            'emotional_weight': self.emotional_weight,
            'access_count': self.access_count,
            'tier': self.tier.value,
            'metadata': self.metadata,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'creation_time': self.creation_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """딕셔너리에서 메모리 엔트리 생성"""
        return cls(
            id=data['id'],
            content=data['content'],
            concepts=data['concepts'],
            importance=data.get('importance', 0.5),
            emotional_weight=data.get('emotional_weight', 0.0),
            access_count=data.get('access_count', 0),
            tier=MemoryTier(data.get('tier', 'working')),
            metadata=data.get('metadata', {}),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            creation_time=datetime.fromisoformat(data['creation_time'])
        )
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.access_count += 1
        self.last_accessed = datetime.now()
