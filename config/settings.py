"""
설정 관리 모듈
전체 시스템의 설정값들을 중앙에서 관리
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SystemConfig:
    """시스템 전반의 설정"""
    # Redis 설정
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_ttl: int = 1800  # 30분
    
    # SQLite 설정
    db_path: str = './memory.db'
    
    # ChromaDB 설정
    chroma_path: str = './chroma_db'
    collection_name: str = 'long_term_memories'
    
    # 메모리 관리 설정
    max_active_memory: int = 1000
    max_session_duration: int = 3600  # 1시간
    
    # 연관 네트워크 설정
    min_connection_strength: float = 0.1
    decay_factor: float = 0.95
    
    # 임베딩 모델 설정
    embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'redis': {
                'host': self.redis_host,
                'port': self.redis_port,
                'ttl': self.redis_ttl
            },
            'storage': {
                'db_path': self.db_path,
                'chroma_path': self.chroma_path,
                'collection_name': self.collection_name
            },
            'memory': {
                'max_active_memory': self.max_active_memory,
                'max_session_duration': self.max_session_duration
            },
            'network': {
                'min_connection_strength': self.min_connection_strength,
                'decay_factor': self.decay_factor
            },
            'embedding': {
                'model': self.embedding_model
            }
        }
