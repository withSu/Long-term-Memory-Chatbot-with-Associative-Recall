"""
메모리 관리 통합 모듈
모든 메모리 계층을 통합하여 관리
"""
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from models.memory_entry import MemoryEntry
from models.enums import MemoryTier
from storage.redis_storage import RedisStorage
from storage.sqlite_storage import SQLiteStorage
from storage.vector_storage import VectorStorage
from utils.cache import LRUCache
from utils.bloom_filter import BloomFilter


class MemoryManager:
    """
    통합 메모리 관리자
    
    기능:
    - 계층화된 메모리 저장소 관리
    - 효율적인 검색 및 저장
    - 비동기 처리 통합
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 저장소 초기화
        self.redis = RedisStorage(
            host=config['redis']['host'],
            port=config['redis']['port'],
            ttl=config['redis']['ttl']
        )
        
        self.sqlite = SQLiteStorage(
            db_path=config['storage']['db_path']
        )
        
        self.vector = VectorStorage(
            db_path=config['storage']['chroma_path'],
            collection_name=config['storage']['collection_name'],
            embedding_model=config['embedding']['model']
        )
        
        # 캐시 및 인덱스
        self.search_cache = LRUCache(capacity=1000)
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)
        
        # 통계
        self.stats = {
            'total_memories': 0,
            'search_operations': 0,
            'save_operations': 0,
            'tier_distribution': {tier: 0 for tier in MemoryTier}
        }
    
    async def save_memory(
        self,
        content: Dict[str, Any],
        concepts: List[str],
        importance: float = 0.5,
        emotional_weight: float = 0.0
    ) -> str:
        """메모리 저장"""
        memory_id = str(uuid.uuid4())
        
        # 메모리 엔트리 생성
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            concepts=concepts,
            importance=importance,
            emotional_weight=emotional_weight,
            tier=MemoryTier.WORKING
        )
        
        # 워킹 메모리에 저장
        await self.redis.save(memory)
        
        # 블룸 필터 업데이트
        for concept in concepts:
            self.bloom_filter.add(concept)
        
        # 통계 업데이트
        self.stats['total_memories'] += 1
        self.stats['save_operations'] += 1
        self.stats['tier_distribution'][MemoryTier.WORKING] += 1
        
        return memory_id
    async def _search_working_memory(self, concepts: List[str]) -> List[MemoryEntry]:
        """워킹 메모리 검색"""
        try:
            return await self.redis.find_by_concepts(concepts)
        except Exception as e:
            print(f"워킹 메모리 검색 오류: {e}")
            return []

    async def _search_short_term(self, concepts: List[str]) -> List[MemoryEntry]:
        """단기 메모리 검색"""
        try:
            return await self.sqlite.find_by_concepts(concepts)
        except Exception as e:
            print(f"단기 메모리 검색 오류: {e}")
            return []

    async def _search_long_term(self, concepts: List[str], query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """장기 메모리 검색"""
        try:
            if query_text:
                return await self.vector.search_by_text(query_text, top_k=10)
            else:
                return await self.vector.search_by_concepts(concepts, top_k=10)
        except Exception as e:
            print(f"장기 메모리 검색 오류: {e}")
            return []

    async def search_memories(
        self,
        concepts: List[str],
        query_text: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """통합 메모리 검색"""
        # 캐시 확인
        cache_key = f"search:{':'.join(concepts)}"
        cached_result = self.search_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 병렬 검색 실행
        search_tasks = [
            self._search_working_memory(concepts),
            self._search_short_term(concepts),
            self._search_long_term(concepts,query_text)
        ]
        
        results = await asyncio.gather(*search_tasks)
        
        # 연관 네트워크 검색 - 이 부분이 없어서 오류가 발생
        network_results = {}
        if hasattr(self, 'network') and self.network:
            network_results = self.network.find_associations(concepts, depth=2)
        else:
            # 네트워크가 없는 경우 빈 딕셔너리 사용
            network_results = {}
        
        # 결과 병합 및 재순위
        merged_results = self._merge_search_results(results, network_results, concepts)
        
        # 캐시에 저장
        self.search_cache.put(cache_key, merged_results)
        
        # 통계 업데이트
        self.stats['search_operations'] += 1
        
        return merged_results
    
    def _merge_search_results(
        self,
        search_results: List[List],
        network_results: Dict,
        query_concepts: List[str]  # 이 매개변수 추가 필요
    ) -> List[Dict]:
        # 로깅 추가
        print(f"Search results: {len(search_results)}")
        for i, result in enumerate(search_results):
            print(f"Result set {i}: {len(result)} items")
        print(f"Network results: {len(network_results)} items")
        all_results = []

        # Redis 결과 (최고 우선순위)
        redis_results = search_results[0]
        for memory in redis_results:
            all_results.append({
                'memory': memory,
                'score': 1.0,
                'source': 'working'
            })
        
         # SQLite 결과
        sqlite_results = search_results[1]
        for memory in sqlite_results:
            # 개념 매칭 점수 계산
            concept_match_score = 0
            if query_concepts:  # 빈 리스트 검사 추가
                concept_match_score = sum(1 for c in memory.concepts if c in query_concepts) / len(query_concepts)
            score = 0.8 * memory.importance + 0.2 * concept_match_score
            all_results.append({
                'memory': memory,
                'score': score,
                'source': 'short_term'
            })
        
        # Vector DB 결과
        vector_results = search_results[2]
        for result in vector_results:
            score = 0.6 * result['similarity']
            all_results.append({
                'memory': result,
                'score': score,
                'source': 'long_term'
            })
        
        # 점수 기반 정렬
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        
        # 중복 제거
        unique_results = []
        seen_ids = set()
        
        for result in sorted_results:
            memory_id = result['memory'].id if hasattr(result['memory'], 'id') else result['memory']['id']
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
        
        return unique_results[:10]  # 상위 10개만 반환
    
    async def get_stats(self) -> Dict[str, Any]:
        """통합 통계 반환"""
        redis_stats = self.redis.get_stats()
        sqlite_stats = self.sqlite.get_stats()
        vector_stats = self.vector.get_stats()
        
        return {
            'total_memories': self.stats['total_memories'],
            'tier_distribution': {
                'working': redis_stats['total_memories'],
                'short_term': sqlite_stats['total_memories'],
                'long_term': vector_stats['total_memories']
            },
            'operations': {
                'search': self.stats['search_operations'],
                'save': self.stats['save_operations']
            },
            'resources': {
                'memory_usage': redis_stats['memory_usage'],
                'cache_size': len(self.search_cache.cache)
            }
        }