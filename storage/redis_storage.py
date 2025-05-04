"""
Redis 저장소 클래스
워킹 메모리 관리를 위한 Redis 인터페이스
"""
import json
import redis
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.memory_entry import MemoryEntry
from models.enums import MemoryTier


class RedisStorage:
    """
    Redis 워킹 메모리 저장소
    
    특징:
    - TTL 기반 자동 만료
    - 빠른 접근 속도
    - 임시 데이터 저장에 최적화
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, ttl: int = 1800):
        self.client = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        self.ttl = ttl  # Time To Live in seconds
    
    async def save(self, memory: MemoryEntry) -> None:
        """메모리 저장"""
        key = f"memory:{memory.id}"
        value = json.dumps(memory.to_dict())
        
        # TTL과 함께 저장
        self.client.setex(key, self.ttl, value)
        
        # 개념별 인덱스 업데이트
        self._update_concept_index(memory)
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """메모리 조회"""
        key = f"memory:{memory_id}"
        value = self.client.get(key)
        
        if value:
            data = json.loads(value)
            return MemoryEntry.from_dict(data)
        return None
    
    async def delete(self, memory_id: str) -> None:
        """메모리 삭제"""
        key = f"memory:{memory_id}"
        self.client.delete(key)
    
    async def find_by_concepts(self, concepts: List[str]) -> List[MemoryEntry]:
        """개념으로 메모리 검색"""
        memories = []
        
        for concept in concepts:
            # 개념 인덱스에서 메모리 ID 조회
            concept_key = f"concept:{concept}"
            memory_ids = self.client.smembers(concept_key)
            
            for memory_id in memory_ids:
                memory = await self.get(memory_id)
                if memory:
                    memories.append(memory)
        
        return memories
    
    def _update_concept_index(self, memory: MemoryEntry) -> None:
        """개념 인덱스 업데이트"""
        for concept in memory.concepts:
            concept_key = f"concept:{concept}"
            self.client.sadd(concept_key, memory.id)
            # 인덱스도 TTL 설정
            self.client.expire(concept_key, self.ttl)
    
    async def save_context(self, user_id: str, context: Dict[str, Any]) -> None:
        """컨텍스트 저장"""
        key = f"context:{user_id}"
        value = json.dumps(context)
        self.client.setex(key, 3600, value)  # 1시간 TTL
    
    async def get_context(self, user_id: str) -> Dict[str, Any]:
        """컨텍스트 조회"""
        key = f"context:{user_id}"
        value = self.client.get(key)
        return json.loads(value) if value else {}
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        memory_keys = self.client.keys("memory:*")
        return {
            'total_memories': len(memory_keys),
            'memory_usage': self.client.info()['used_memory'],
            'connected_clients': self.client.info()['connected_clients']
        }
